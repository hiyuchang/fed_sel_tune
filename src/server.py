import os
import sys
import torch
import pickle
import random
from scipy.stats import truncnorm
from collections import OrderedDict
from path import Path
_CURRENT_DIR = Path(__file__).parent.abspath()
sys.path.append(_CURRENT_DIR.parent)
sys.path.append(_CURRENT_DIR.parent / "data")

from client import Client
from helpers.clip_cifar10 import CLIP_CIFAR10
from helpers.args import get_args
from helpers.util import (
    LOG_DIR,
    TEMP_DIR,
    MODEL_PATH,
    get_prefix,
    fix_random_seed,
    clone_parameters,
    find_selection_set,
    get_logger,
    get_client_data_size,
)


class Server:
    def __init__(self, args):
        self.args = args
        self.model_type = args.model_type
        self.dataset = args.dataset
        self.strategy = args.strategy
        self.n_layers = args.n_layers
        self.device = torch.device(
            "cuda:" + str(self.args.gpu)
            if self.args.gpu >= 0 and torch.cuda.is_available()
            else "cpu"
        )

        self.client_num_in_total = self.args.client_num_in_total
        self.client_num_per_round = self.args.client_num_per_round
        self.client_id_indices = list(range(self.client_num_in_total))
        self.datasize_dict = get_client_data_size(self.dataset, self.args)

        ##### get run_name and logger #####
        if self.args.n_layers_inc != 0:
            layer_text = f"{self.args.n_layers}+{self.args.n_layers_inc}layer"
        else:
            layer_text = f"{self.args.n_layers}layer"
        if self.strategy == "pro":
            self.pre_text = f"pro{self.args.balance}_{self.args.metric}_{layer_text}{self.args.sel_gap, self.args.sel_batch}"
        else:
            self.pre_text = f"{self.strategy}_{layer_text}"
        
        self.temp_dir = (
            TEMP_DIR / f"{self.model_type}_{self.dataset}" / self.pre_text
        )
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)
        run_name = (
            self.pre_text
            + f"_C{self.args.client_num_per_round}of{self.client_num_in_total}_alpha{self.args.alpha}"
        )
        self.log_name = run_name + ".log"
        self.logger = get_logger(
            name="default_log", log_name=self.log_name, save_log=self.args.save_log
        )

        ##### create model #####
        if self.model_type == "clip":
            import clip
            clip_model, preprocess = clip.load("ViT-B/32", device=self.device)
            _dummy_model = CLIP_CIFAR10(
                clip_model.visual, num_classes=self.args.num_classes
            )
            self.args.preprocess = preprocess
        elif "bert" in self.model_type:
            from transformers import AutoModelForSequenceClassification
            _dummy_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH[self.model_type], num_labels=self.args.num_classes
            )
        else:
            raise NotImplementedError(f"Model {self.model_type} is not implemented.")

        ##### initialize global model #####
        passed_epoch = 0
        self.global_params_dict = None
        if self.args.resume and os.listdir(self.temp_dir) != []:
            if os.path.exists(self.temp_dir / "global_model.pt"):
                self.global_params_dict = torch.load(self.temp_dir / "global_model.pt")
                self.logger.info("Find existing global model...")
            else:
                self.logger.info("Not found global model...")
                exit()
        else:
            self.global_params_dict = OrderedDict(_dummy_model.state_dict())
        for name, param in self.global_params_dict.items():
            self.global_params_dict[name] = param.to(self.device)
        self.global_epochs = self.args.global_epochs - passed_epoch

        ##### get layer and parameter names #####
        self.all_params_name = [name for name in _dummy_model.state_dict().keys()]
        self.trainable_params_name = [
            name
            for name, param in _dummy_model.state_dict(keep_vars=True).items()
            if param.requires_grad
        ]
        self.layer_to_name_dict, self.name_to_layer_dict = get_prefix(
            _dummy_model, self.args.model_type
        )
        self.trainable_layers = [
            layer
            for layer in self.layer_to_name_dict.keys()
            if layer not in self.args.fixed_layers
        ]
        self.layers_index = {
            layer: i for i, layer in enumerate(list(self.layer_to_name_dict.keys()))
        }
        self.logger.info(f"Layers and index are {self.layers_index}")
        self.args.layers_index = self.layers_index

        ##### fix embedding layer #####
        for name in self.args.fixed_names:
            if name in self.trainable_params_name:
                self.trainable_params_name.remove(name)
            _dummy_model.state_dict()[name].requires_grad = False

        ##### get layer budgets #####
        self.args.n_layers_list = [self.args.n_layers] * self.args.client_num_in_total
        if self.args.n_layers_inc != 0:  # NOTE: half normal distribution
            lower, upper = self.args.n_layers - 1, self.args.n_layers_inc + 1
            tmp = truncnorm(a=lower, b=upper, loc=1, scale=1).rvs(
                size=self.args.client_num_in_total
            )
            self.args.n_layers_list = [round(x) for x in tmp]

        ##### initialize client #####
        self.trainer = Client(
            backbone=_dummy_model,
            args=self.args,
            logger=self.logger,
            trainable_layers=self.trainable_layers,
            layer_to_name_dict=self.layer_to_name_dict,
            name_to_layer_dict=self.name_to_layer_dict
        )
        
        ##### initialize dicts #####
        self.layer_to_client_dict = {} # Each layer has some associated clients
        self.client_to_layer_dict = OrderedDict(
            {i: [] for i in range(self.client_num_in_total)}
        ) # Each client has some selected layers
        self.selected_clients = self.client_id_indices
        
        fix_random_seed(self.args.seed)

    def train(self):
        self.logger.info("=" * 30 + "TRAINING" + "=" * 30)
        self.result = {}

        for E in range(1, self.global_epochs + 1):
            self.current_epoch = E
            self.result[E] = {}
            self.logger.info("=" * 30 + f"ROUND: {E}" + "=" * 30)

            ##### select clients #####
            selected_clients = random.sample(
                self.client_id_indices,
                min(self.args.client_num_per_round, self.args.client_num_in_total),
            )
            self.selected_clients = sorted(selected_clients)
            self.logger.info(f"Selected clients: {self.selected_clients}")
            self.logger.info(f"Budgets: {[self.args.n_layers_list[i] for i in self.selected_clients]}")

            ##### select layers #####
            self.client_to_layer_dict = self.select_layers(self.selected_clients, E)
            self.layer_to_client_dict = {
                layer: [client_id for client_id in self.selected_clients if layer in self.client_to_layer_dict[client_id]]
                for layer in self.trainable_layers
            } # inlude common layers

            ##### clients locally train #####
            updated_params = OrderedDict()
            for client_id in self.selected_clients:
                client_local_params = clone_parameters(self.global_params_dict, to_cpu=True)

                result = self.trainer.train(
                    client_id=client_id,
                    model_params=client_local_params,
                    current_epoch=E,
                    selected_layers=self.client_to_layer_dict[client_id],
                )

                new_params = result["selected_params"]
                updated_params[client_id] = clone_parameters(new_params, to_cpu=True) # these are gradients

            ##### aggregation #####
            self.aggregation(self.selected_clients, updated_params)

            ##### test periodically #####
            if E % self.args.save_period == 0:
                self.test(E)  # accuracy of all clients

            ##### save model periodically #####
            if self.args.save_model and E % self.args.save_period == 0:
                torch.save(
                    self.global_params_dict,
                    self.temp_dir / "global_model.pt",
                )
                with open(self.temp_dir / "epoch.pkl", "wb") as f:
                    pickle.dump(E, f)
        return

    @torch.no_grad()
    def test(self, E=None):
        if self.args.only_train:
            return

        self.logger.info("=" * 30 + "TESTING" + "=" * 30)
        stats = self.trainer.test(
            client_id=0,
            model_params=self.global_params_dict,
        )
        loss = stats["loss"]
        acc = stats["acc"]

        self.logger.info("=" * 20 + "RESULTS" + "=" * 20)
        self.logger.info(
            "Global loss: {:.4f}    Global accuracy: {:.2f}%".format(
                loss,
                acc,
            )
        )

        E = self.args.global_epochs if E is None else E
        if E == 0:
            self.best_dict = {"Best/Acc": acc, "Best/Loss": loss}
        else:
            self.best_dict["Best/Acc"] = max(self.best_dict["Best/Acc"], acc)
            self.best_dict["Best/Loss"] = min(self.best_dict["Best/Loss"], loss)

        return

    @torch.no_grad()
    def aggregation(self, selected_clients, updated_params):
        self.logger.info("Aggregating models")
        grad_avg = OrderedDict()

        # only selected layers are aggregated
        changed_name = []
        key_list = list(updated_params[selected_clients[0]].keys())
        trainable_params_name = [x for x in self.trainable_params_name if x in key_list]

        for name in trainable_params_name:
            layer = self.name_to_layer_dict[name]
            if layer in self.layer_to_client_dict:
                eff_clients = [
                    idx
                    for idx in self.layer_to_client_dict[layer]
                    if idx in selected_clients
                ]
                a = self.datasize_dict
                a_sum = sum([a[idx] for idx in eff_clients])

                grad_avg[name] = torch.zeros_like(self.global_params_dict[name]).cpu()
                for client_id in eff_clients:
                    grad_avg[name] += (
                        updated_params[client_id][name] * a[client_id] / a_sum
                    )
                changed_name.append(name)

                grad_avg[name] = grad_avg[name].to(self.device)
                self.global_params_dict[name].sub_(grad_avg[name] * self.args.local_lr)
        
        # self.logger.debug(f"Changed names are {changed_name}")
        del updated_params
        
        return

    def select_layers(self, selected_clients, current_epoch):
        if self.strategy == "full":
            from copy import deepcopy
            self.client_to_layer_dict = {client_id: deepcopy(self.trainable_layers) for client_id in selected_clients}
            return self.client_to_layer_dict

        elif self.strategy in ("top", "bottom", "both"):
            if self.strategy == "top":  # output layer
                layer_list = list(reversed(self.trainable_layers))
            elif self.strategy == "bottom":  # input layer
                layer_list = self.trainable_layers
            elif self.strategy == "both":
                layer_list = [None] * len(self.trainable_layers) * 2
                layer_list[::2] = self.trainable_layers
                layer_list[1::2] = list(reversed(self.trainable_layers))

            for layer in self.args.common_layers:
                if layer in layer_list:
                    layer_list.remove(layer)
            selected_layers = layer_list[:n_layers]

            ##### add common layers #####
            selected_layers.extend(self.args.common_layers)
            self.client_to_layer_dict = {client_id: selected_layers for client_id in selected_clients}
            return self.client_to_layer_dict
        
        sel_gap = (
            self.args.global_epochs if self.args.sel_gap == -1 else self.args.sel_gap
        )
        ##### NOT reach period: use previous selection #####
        if (current_epoch - 1) % sel_gap != 0:
            for client_id in selected_clients:
                if self.client_to_layer_dict[client_id] == []:
                    n_layers = self.args.n_layers_list[i]
                    self.client_to_layer_dict[client_id] = self.selected_layers_global[
                        :n_layers
                    ]
                    # extend common layers to avoid missing
                    if (
                        self.args.common_layers[0]
                        not in self.client_to_layer_dict[client_id]
                    ):
                        self.client_to_layer_dict[client_id].extend(
                            self.args.common_layers
                        )
                # print(client_id, self.client_to_layer_dict[client_id])
            return self.client_to_layer_dict
        
        ##### reach period: select layers #####
        norms_matrix = []
        glb_grad_norm = {}
        a = self.datasize_dict
        a_sum = sum([a[idx] for idx in selected_clients])

        # all metrics
        for client_id in selected_clients:
            client = self.trainer
            client.client_id = client_id
            client.model.load_state_dict(self.global_params_dict, strict=True)
            client.get_client_local_dataset()
            client.model.to(self.device)

            grad_norm = client.init_train()
            norms_matrix.append(list(grad_norm.values()))

        if self.strategy == "pro":
            if glb_grad_norm == {}:
                for layer, norm2 in grad_norm.items():
                    glb_grad_norm[layer] = norm2 * (a[client_id] / a_sum)
            else:
                for layer, norm2 in grad_norm.items():
                    glb_grad_norm[layer] += norm2 * a[client_id] / a_sum

            for layer in self.args.common_layers:
                for name in self.layer_to_name_dict[layer]:
                    if name in grad_norm:
                        grad_norm.pop(name)

            ##### select layers #####
            costs = [1] * len(norms_matrix[0])
            budgets = [self.args.n_layers_list[i] for i in selected_clients]
            last_selected = []
            for i in range(len(selected_clients)):
                client_id = selected_clients[i]
                last_selected.append(
                    [
                        self.layers_index[x]
                        for x in self.client_to_layer_dict[client_id]
                    ]
                )
            if current_epoch == 1:
                last_selected = [[12] for _ in range(len(selected_clients))] # initilization

            selected_layers = find_selection_set(
                norms_matrix,
                costs,
                budgets,
                last_selected,
                self.current_epoch,
                self.args,
            )

            for i, client_id in enumerate(selected_clients):
                layer_indices = selected_layers[i]
                self.client_to_layer_dict[client_id] = [
                    self.trainable_layers[x] for x in layer_indices
                ]
                self.client_to_layer_dict[client_id].extend(self.args.common_layers)

            return self.client_to_layer_dict

        elif self.strategy in ("rgn", "snr"):
            for idx, client_id in enumerate(selected_clients):
                grad_norm = norms_matrix[idx]
                layer_list = list(
                    sorted(grad_norm.keys(), key=lambda x: grad_norm[x], reverse=True)
                )
                for layer in self.args.common_layers:
                    if layer in layer_list:
                        layer_list.remove(layer)
                selected_layers = layer_list[:n_layers]
                selected_layers.extend(self.args.common_layers)
                self.client_to_layer_dict[client_id] = selected_layers
            return self.client_to_layer_dict
        
        else:
            raise NotImplementedError(f"Strategy {self.strategy} is not implemented.")

    def run(self):
        if self.args.only_test:
            with open(self.temp_dir / "epoch.pkl", "rb") as f:
                current_epoch = pickle.load(f)
                current_epoch = int(current_epoch)
            self.global_params_dict = torch.load(self.temp_dir / "global_model.pt")

            self.test(current_epoch)

            # save results
            with open(self.temp_dir / "result.pkl", "wb") as f:
                result = {
                    "epoch": current_epoch,
                    "result": self.result,
                    "args": self.args,
                }
                pickle.dump(result, f)
            return

        if self.args.save_log:
            if not os.path.isdir(LOG_DIR):
                os.mkdir(LOG_DIR)
            with open(LOG_DIR / self.log_name, "w") as file:
                file.truncate(0)

        self.test(0) # before training
        self.train()
        self.test() # after training
        return

if __name__ == "__main__":
    args = get_args()
    server = Server(args)
    server.run()

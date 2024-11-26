import torch
from collections import OrderedDict
from typing import OrderedDict
from copy import deepcopy

from helpers.util import get_dataloader


class Client:
    def __init__(
        self,
        backbone,
        args,
        logger,
        trainable_layers,
        layer_to_name_dict,
        name_to_layer_dict
    ):
        ###### basic setup ######
        self.device = torch.device(
            "cuda:" + str(args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
        )
        self.args = args
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.local_epochs = args.local_epochs
        self.local_steps = args.local_steps
        self.local_lr = args.local_lr
        self.dataset = args.dataset
        self.n_layers = args.n_layers
        self.strategy = args.strategy
        self.sel_batch = args.sel_batch
        self.metric = args.metric
        self.model_type = args.model_type
        self.img_dtype = args.img_dtype
        self.logger = logger
        
        self.trainable_layers = trainable_layers
        self.layer_to_name_dict = layer_to_name_dict
        self.name_to_layer_dict = name_to_layer_dict
        
        self.client_id = None
        self.model = deepcopy(backbone)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def test(
        self,
        client_id,
        model_params,
    ):
        self.client_id = client_id
        self.model.load_state_dict(model_params)
        self.get_client_local_dataset()

        loss, acc = self.evaluate()
        
        stats = {"loss": loss, "acc": acc}
        
        return stats

    def train(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
        current_epoch=None,
        selected_layers=None,
    ):
        self.client_id = client_id
        self.model.load_state_dict(model_params)
        self.get_client_local_dataset()

        self.model.to(self.device)

        ###### adjust learning rate ######
        if self.args.lr_decay != 1:
            local_lr = (
                self.args.lr_decay ** ((current_epoch - 1) // self.args.lr_decay_step)
                * self.local_lr
            )
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=local_lr)

        ###### train ######
        res_dict = self._train_model(selected_layers)

        return res_dict

    def _train_model(self, selected_layers):
        self.model.train()
        self.selected_layers = selected_layers

        if self.strategy == "full":
            self.logger.info(f"Tune all {len(self.selected_layers)} layers")
        else:
            self.logger.info(f"Only tune {self.selected_layers} while freezing other layers")

        selected_name_list = []
        for layer in self.selected_layers:
            selected_name_list.extend(self.layer_to_name_dict[layer])
        # print("selected_name_list",  selected_name_list)

        for name, param in self.model.named_parameters():
            if name in selected_name_list:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        ###### either local_epochs or local_steps works ######
        if self.local_steps == 0:
            local_epochs = self.local_epochs
            target_steps = len(self.trainset)
        else:
            local_epochs = 1
            target_steps = self.local_steps

        ###### train for local_epochs ######
        gradients = {}
        for _ in range(local_epochs):
            kwargs = {"recorded_names": selected_name_list, "gradients": gradients}
            gradients, train_loss, train_acc = self._forward_pass(
                self.trainset,
                self.device,
                target_steps,
                keys_to_return=["step", "grad"],
                **kwargs,
            )

        # only return the difference of the selected layers
        selected_params = OrderedDict()
        for name, param in self.model.named_parameters():
            if name in selected_name_list and name in gradients:
                selected_params[name] = gradients[name]

        return {
            "selected_params": selected_params,
            "train_loss": train_loss,
            "train_acc": train_acc,
        }

    def init_train(self):
        self.model.train()
        num_batch = (
            len(self.trainset) if self.sel_batch == -1 else self.sel_batch
        )
        self.optimizer.zero_grad()

        ##### fix certain layers #####
        recorded_names = []
        for name, param in self.model.named_parameters():
            if name not in self.args.fixed_names:
                param.requires_grad_(True)
                recorded_names.append(name)

        ##### initialize grad_norm and param_norm #####
        grad_norm, param_norm = OrderedDict(), OrderedDict()
        for layer in self.trainable_layers:
            if layer in self.args.fixed_layers or layer in self.args.common_layers:
                continue  # skip fixed layers
            grad_norm[layer] = 0.0
            param_norm[layer] = 0.0

        ###### train whole model once and get values ######
        kwargs = {
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "recorded_names": recorded_names,
            "gradients": {},
        }
        if self.metric == "diff" or self.metric == "rgn":
            (grad_norm, param_norm) = self._forward_pass(
                self.trainset, self.device, num_batch, keys_to_return=["norm"], **kwargs
            )
            if self.metric == "rgn":
                for layer in grad_norm.keys():
                    grad_norm[layer] = grad_norm[layer] / param_norm[layer]
            
            return OrderedDict(grad_norm)
        else:
            gradients, _, _ = self._forward_pass(
                self.trainset, self.device, num_batch, keys_to_return=["grad"], **kwargs
            )

            snr = OrderedDict()
            for name, grad in gradients.items():
                layer = self.name_to_layer_dict[name]
                if layer not in snr:
                    snr[layer] = 0.0
                snr[layer] += torch.mean(grad).item() ** 2 / torch.std(grad).item()
            return snr

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        self.model.to(self.device)

        if self.model_type == "clip":
            size, loss, correct = 0, 0, 0
            for images, labels in self.testset:
                images = images.to(self.device, dtype=self.img_dtype)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
                size += len(labels)

            acc = correct / size * 100.0
            loss = loss / len(self.testset)
            return loss, acc

        else:
            size = 0
            loss = 0
            correct = 0
            for batch in self.testset:
                input_ids = batch["input_ids"].to(self.device)  # torch.Size([64, 256])
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss += self.criterion(logits, labels).item()
                pred = torch.softmax(logits, -1).argmax(-1)
                correct += (pred == labels).int().sum()
                size += labels.size(-1)

            acc = correct / size * 100.0
            loss = loss / len(self.testset)
            return loss, acc

    def get_client_local_dataset(self):
        datasets = get_dataloader(
            self.dataset,
            self.client_id,
            preprocess=self.args.preprocess,
            args=self.args,
            only_dataset=False,
        )

        self.trainset = datasets["train"]
        self.testset = datasets["test"]
        self.datasize = len(self.trainset.dataset)
        return

    def _forward_pass(self, loader, device, target_step, keys_to_return, **kwargs):
        ##### fix certain layers #####
        for name, param in self.model.named_parameters():
            if name in self.args.fixed_names:
                param.requires_grad_(False)

        if "norm" in keys_to_return:
            grad_norm = kwargs["grad_norm"]
            param_norm = kwargs["param_norm"]
        if "grad" in keys_to_return:
            recorded_names = kwargs["recorded_names"]
            gradients = kwargs["gradients"]

        ##### forward #####
        loss_tot, correct, size = 0, 0, 0
        step = 0
        for batch in loader:
            if self.model_type == "clip":
                images, labels = batch
                images = images.to(self.device, dtype=self.img_dtype)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            else:
                input_ids = batch["input_ids"].to(
                    self.device
                ) 
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = self.criterion(logits, labels)

            loss_tot += loss.item()
            pred = torch.softmax(logits, -1).argmax(-1)
            correct += (pred == labels).int().sum()
            size += labels.size(-1)

            loss.backward()

            ##### record keys #####
            if "norm" in keys_to_return:
                # NOTE: not stepping here
                for name, param in self.model.named_parameters():
                    layer = self.name_to_layer_dict[name]
                    if (layer in self.trainable_layers) and (
                        layer not in self.args.common_layers
                    ):
                        grad_norm[layer] += param.grad.norm().item() ** 2
                        param_norm[layer] += param.norm().item() ** 2
            if "grad" in keys_to_return:
                for name, param in self.model.named_parameters():
                    if name in recorded_names:
                        if name not in gradients:
                            gradients[name] = param.grad.clone().detach()
                        else:
                            gradients[name] += param.grad.clone().detach()
                self.optimizer.zero_grad()

            ##### step if necessary #####
            if "step" in keys_to_return:
                self.optimizer.step()
                self.optimizer.zero_grad()

            step += 1
            if target_step > 0 and step >= target_step:
                break

        acc = correct / size * 100.0
        loss_tot = loss_tot / len(loader)
        
        self.logger.info(
            "client {} local training  loss: {:.4f}  accuracy: {:.2f}%".format(
                self.client_id, loss_tot, acc
            )
        )

        if "norm" in keys_to_return:
            return grad_norm, param_norm
        if "grad" in keys_to_return:
            return gradients, loss_tot, acc
        return loss_tot, acc

        # ##### BERT model #####
        # if "norm" in keys_to_return:
        #     grad_norm = kwargs["grad_norm"]
        #     param_norm = kwargs["param_norm"]

        #     for x, y in loader:
        #         x, y = x.to(device), y.to(device)
        #         if output_hidden_states:
        #             logits, hidden_states = self.model(x, output_hidden_states=True)
        #         else:
        #             logits = self.model(x)
        #         loss = self.criterion(logits, y)
        #         self.optimizer.zero_grad()
        #         loss.backward()

        #         for name, param in self.model.named_parameters():
        #             if "bn" in name:
        #                 continue
        #             layer = self.name_to_layer_dict[name]
        #             if (
        #                 layer in self.args.common_layers
        #                 or layer in self.args.fixed_layers
        #             ):
        #                 continue
        #             if layer in self.trainable_layers:
        #                 grad_norm[layer] += param.grad.norm().item()
        #                 param_norm[layer] += param.norm().item()

        #         step += 1
        #         if target_step > 0 and step >= target_step:
        #             break

        #     if "norm" in keys_to_return:
        #         return grad_norm, param_norm

        # # return gradients
        # recorded_names = kwargs["recorded_names"]
        # gradients = {}

        # step = 0
        # loss_tot, correct, size = 0, 0, 0
        # for x, y in loader:
        #     x, y = x.to(device), y.to(device)
        #     logits = self.model(x)
        #     loss = self.criterion(logits, y)
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     loss_tot += loss.item()

        #     # compute accuracy
        #     pred = torch.argmax(logits, dim=-1)  # already masked
        #     correct += torch.sum(pred == y).item()
        #     size += y.size(0)

        #     for name, param in self.model.named_parameters():
        #         if name in recorded_names:
        #             if name not in gradients:
        #                 gradients[name] = param.grad.clone().detach()
        #             else:
        #                 gradients[name] += param.grad.clone().detach()

        #     self.optimizer.step()
        #     step += 1
        #     if target_step > 0 and step >= target_step:
        #         break

        # loss_tot = loss_tot / len(loader)
        # acc = correct / size * 100.0
        # return gradients, loss_tot, acc

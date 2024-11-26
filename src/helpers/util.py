import json
import pickle
import random
import clip
import torch
import numpy as np
import cvxpy as cp
import logging
from collections import OrderedDict
from path import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLayer
from transformers.models.roberta.modeling_roberta import RobertaLayer

from data_helpers.domain import prepare_domain_data
from data_helpers.nlp import partition_multi_lang_data
from data_helpers.dirichlet import dirichlet_dict

PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()
LOG_DIR = PROJECT_DIR / "logs"
TEMP_DIR = PROJECT_DIR / "temp"
MAIN_DIR = Path(__file__).parent.parent.parent.parent.abspath()
DATASETS_DIR = MAIN_DIR / "datasets"
MODEL_DIR = MAIN_DIR / "models"
MODEL_PATH = {
    "xlm-roberta-large": MODEL_DIR / "xlm-roberta-large",
    "roberta-large": MODEL_DIR / "roberta-large",
    "xlm-roberta-base": MODEL_DIR / "xlm-roberta-base",
    "roberta-base": MODEL_DIR / "roberta-base",
}
DOMAINNET_SITE = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]


def get_dataloader(
    dataset,
    client_id,
    preprocess,
    args,
    only_dataset=False,
):
    batch_size = args.batch_size
    if dataset == "domainnet":
        trainset, testset = prepare_domain_data(
            args, only_size=False, client_id=client_id
        )
        if only_dataset:
            return {"train": trainset, "test": testset}
        trainloader = DataLoader(trainset, batch_size, pin_memory=True, shuffle=True)
        testloader = DataLoader(testset, batch_size * 2, pin_memory=True)
        return {"train": trainloader, "test": testloader}

    elif dataset == "xglue":
        trainset_dict, testset, dict_users = partition_multi_lang_data(args.model_type, n_clients=args.client_num_in_total)
        trainset = trainset_dict[client_id]
        if only_dataset:
            return {"train": trainset, "test": testset}
        trainloader = DataLoader(trainset, batch_size, pin_memory=True, shuffle=True)
        testloader = DataLoader(testset, batch_size * 2, pin_memory=True)
        return {"train": trainloader, "test": testloader}

    elif dataset == "cifar10":
        data_dir = DATASETS_DIR / "cifar10"
        trainset = CIFAR10(
            root=data_dir, transform=preprocess, train=True, download=False
        )
        testset = CIFAR10(root=data_dir, transform=preprocess, train=False)

        if only_dataset:
            return {"train": trainset, "test": testset}
        trainloader = DataLoader(trainset, batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size * 2)
        return {"train": trainloader, "test": testloader}

    # client_str = str(client_num_in_total)
    # alpha_str = str(alpha).replace(".", "_")
    # args_dict = json.load(
    #     open(DATASETS_DIR / dataset / client_str / alpha_str / "args.json", "r")
    # )
    # client_num_in_each_pickles = args_dict["client_num_in_each_pickles"]
    # if args.model_type == "clip":
    #     pickles_dir = (
    #         DATASETS_DIR / dataset / "clip" / client_str / alpha_str / "pickles"
    #     )
    # else:
    #     pickles_dir = DATASETS_DIR / dataset / client_str / alpha_str / "pickles"
    # if os.path.isdir(pickles_dir) is False:
    #     raise RuntimeError("Please preprocess and create pickles first.")

    # pickle_path = (
    #     pickles_dir / f"{math.floor(client_id / client_num_in_each_pickles)}.pkl"
    # )
    # with open(pickle_path, "rb") as f:
    #     subset = pickle.load(f)
    # client_dataset = subset[client_id % client_num_in_each_pickles]

    # val_samples_num = 0  # NOTE: no validation
    # train_samples_num = len(client_dataset) - val_samples_num
    # trainset, valset = random_split(
    #     client_dataset,
    #     [train_samples_num, val_samples_num],
    #     generator=torch.Generator().manual_seed(seed),
    # )

    # with open(pickles_dir / f"testdata.pkl", "rb") as f:
    #     testset = pickle.load(f)

    # if only_dataset:
    #     return {"train": trainset, "val": valset, "test": testset}
    # trainloader = DataLoader(trainset, batch_size, shuffle=True)
    # valloader = DataLoader(valset, batch_size)
    # testloader = DataLoader(testset, batch_size * 2)
    # return {"train": trainloader, "val": valloader, "test": testloader}


def get_client_data_size(dataset, args):
    if dataset == "domainnet":
        datasize_dict = prepare_domain_data(args, only_size=True)
        return datasize_dict

    elif dataset == "cifar10":
        dataset_tmp = CIFAR10(root=DATASETS_DIR / "cifar10", train=True, download=True)
        dict_users = dirichlet_dict(
            dataset_tmp, args.client_num_in_total, args.alpha, args.num_classes
        )
        datasize_dict = [len(x) for x in dict_users.values()]
        with open(DATASETS_DIR / dataset / "clip" / "dict_users.json", "w") as f:
            json.dump(dict_users, f)
        return datasize_dict

    elif dataset == "xglue":
        datasize_dict = partition_multi_lang_data(MODEL_PATH[args.model_type], n_clients=args.client_num_in_total, only_size=True)
        return datasize_dict
    
    else:
        ##### load seperation #####
        client_str = str(args.client_num_in_total)
        alpha_str = str(args.alpha).replace(".", "_")
        dataset_pickles_path = (
            DATASETS_DIR / dataset / client_str / alpha_str / "pickles"
        )
        with open(dataset_pickles_path / f"seperation.pkl", "rb") as f:
            seperation = pickle.load(f)

        datasize_dict = OrderedDict()
        for client_id in range(seperation["total"]):
            datasets = get_dataloader(
                dataset,
                client_id,
                preprocess=args.preprocess,
                args=args,
                only_dataset=True,
            )
            datasize = len(datasets["train"])
            datasize_dict[client_id] = datasize

        return datasize_dict


def fix_random_seed(seed):
    torch.cuda.empty_cache()
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def clone_parameters(src, to_cpu=False):
    if to_cpu:
        if isinstance(src, OrderedDict):
            return OrderedDict(
                {
                    name: param.cpu()
                    .clone()
                    .detach()
                    .requires_grad_(param.requires_grad)
                    for name, param in src.items()
                }
            )
        if isinstance(src, torch.nn.Module):
            return OrderedDict(
                {
                    name: param.cpu()
                    .clone()
                    .detach()
                    .requires_grad_(param.requires_grad)
                    for name, param in src.state_dict(keep_vars=True).items()
                }
            )

    if isinstance(src, OrderedDict):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.items()
            }
        )
    if isinstance(src, torch.nn.Module):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.state_dict(keep_vars=True).items()
            }
        )


def get_prefix(_dummy_model, model_type):
    layer_to_name_dict, name_to_layer_dict = OrderedDict(), OrderedDict()
    trainable_params_name = [
        name
        for name, param in _dummy_model.state_dict(keep_vars=True).items()
        if param.requires_grad
    ]

    if model_type == "clip":
        for name, param in _dummy_model.named_modules():
            if isinstance(param, clip.model.ResidualAttentionBlock):
                layer_to_name_dict[name] = []
        layer_to_name_dict["clip.conv1"] = []
        layer_to_name_dict["embedding"] = [
            "clip.class_embedding",
            "clip.positional_embedding",
        ]
        layer_to_name_dict["clip.ln_pre"] = []
        layer_to_name_dict["clip.ln_post"] = []
        layer_to_name_dict["clip.proj"] = ["clip.proj"]
        layer_to_name_dict["classifier"] = []

        # add params for this layer
        for name in trainable_params_name:
            for layer in layer_to_name_dict.keys():
                prefix = layer + "."
                if name.startswith(prefix) and name != layer:
                    layer_to_name_dict[layer].append(name)
                    name_to_layer_dict[name] = layer
        name_to_layer_dict["clip.class_embedding"] = "embedding"
        name_to_layer_dict["clip.positional_embedding"] = "embedding"
        name_to_layer_dict["clip.proj"] = "clip.proj"

    elif model_type in ["xlm-roberta-large", "xlm-roberta-base"]:
        # create layers
        for name, param in _dummy_model.named_modules():
            if isinstance(param, XLMRobertaLayer):
                layer_to_name_dict[name] = []
        layer_to_name_dict["roberta.embeddings"] = []
        layer_to_name_dict["classifier"] = []

        # add params for this layer
        for name in trainable_params_name:
            for layer in layer_to_name_dict.keys():
                prefix = layer + "."
                if name.startswith(prefix) and name != layer:
                    layer_to_name_dict[layer].append(name)
                    name_to_layer_dict[name] = layer

    elif model_type in ["roberta-large", "roberta-base"]:
        # create layers
        for name, param in _dummy_model.named_modules():
            if isinstance(param, RobertaLayer):
                layer_to_name_dict[name] = []
        layer_to_name_dict["roberta.embeddings"] = []
        layer_to_name_dict["classifier"] = []

        # add params for this layer
        for name in trainable_params_name:
            for layer in layer_to_name_dict.keys():
                prefix = layer + "."
                if name.startswith(prefix) and name != layer:
                    layer_to_name_dict[layer].append(name)
                    name_to_layer_dict[name] = layer

    else:
        raise NotImplementedError(f"model type {model_type} is not supported yet")

    # print("name:", len(name_to_layer_dict), "layer:", len(layer_to_name_dict))
    return layer_to_name_dict, name_to_layer_dict


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def find_selection_set(norms, costs, budgets, last_selected, current_epoch, args):
    N = args.client_num_per_round
    K = len(norms[0])
    weight = args.balance
    variables = cp.Variable(N * K, boolean=True)

    it = 0
    while variables.value is None:
        it += 1
        obj = cp.sum(
            [variables[i * K + j] * norms[i][j] for i in range(N) for j in range(K)]
        )
        reg1 = cp.sum(
            [
                (variables[i1 * K + j] - variables[i2 * K + j]) ** 2
                for j in range(K)
                for i1 in range(N)
                for i2 in range(i1 + 1, N)
            ]
        )
        objective = cp.Maximize(obj - weight * reg1)

        budget_constraints = [
            cp.sum([variables[i * K + j] * costs[j] for j in range(K)]) - budgets[i]
            <= 0
            for i in range(N)
        ]

        problem = cp.Problem(objective, budget_constraints)
        problem.solve(solver="ECOS_BB")

        norm_first = [
            np.argsort(np.array(x))[-budgets[i] :].tolist() for i, x in enumerate(norms)
        ]

        if variables.value is not None:
            # transform back
            selected = [[] for i in range(N)]
            for i in range(N):
                cost_i = 0
                for j in range(K):
                    if abs(variables.value[i * K + j] - 1) <= 1e-5:
                        selected[i].append(j)
                        cost_i += costs[j]
                if cost_i < budgets[i]:
                    for j in norm_first[i]:
                        if j not in selected[i]:
                            selected[i].append(j)
                            cost_i += costs[j]
                            if cost_i > budgets[i]:
                                break

            variables = variables.value
            break
        
        ##### avoid infinite loop #####
        elif it == 50: 
            selected = norm_first
            break

    return selected


def get_logger(name, log_name, save_log=False):
    ##### remove root handlers #####
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # lowest level

    ##### print to screen #####
    f1 = logging.Formatter(
        "[%(asctime)s] [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)",
        "%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(f1)
    logger.addHandler(sh)

    ##### save to file #####
    if save_log:
        f2 = logging.Formatter(
            "[%(asctime)s] [%(levelname)8s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        fh = logging.FileHandler(LOG_DIR / log_name)  # encoding='utf8'
        fh.setLevel(logging.INFO)
        fh.setFormatter(f2)
        logger.addHandler(fh)

    return logger

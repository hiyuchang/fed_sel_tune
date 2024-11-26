from path import Path
_CURRENT_DIR = Path(__file__).parent.abspath()
import sys
sys.path.append(_CURRENT_DIR.parent)

import json
import os
import pickle
import random
import numpy as np
import torch
from argparse import ArgumentParser
from torchvision import transforms
from torchvision.datasets import CIFAR10

from dataset import CIFARDataset
from data_helpers import dirichlet_distribution
from helpers.util import DATASETS_DIR

import clip

DATASET = {
    "cifar10": (CIFAR10, CIFARDataset),
}

MEAN = {
    "cifar10": (0.4914, 0.4822, 0.4465),
}

STD = {
    "cifar10": (0.2023, 0.1994, 0.2010),
}


def main(args):
    dataset_root = (
        Path(args.root).abspath() / args.dataset
        if args.root is not None
        else DATASETS_DIR / args.dataset
    )
    client_str = str(args.client_num_in_total)
    alpha_str = str(args.alpha).replace(".", "_")
    if args.clip:
        pickles_dir = (
            DATASETS_DIR / args.dataset / "clip" / client_str / alpha_str / "pickles"
        )
    else:
        pickles_dir = DATASETS_DIR / args.dataset / client_str / alpha_str / "pickles"

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose(
        [transforms.Normalize(MEAN[args.dataset], STD[args.dataset])]
    )
    target_transform = None

    if args.clip:
        os.makedirs(dataset_root / "clip" / client_str / alpha_str, exist_ok=True)
    else:
        os.makedirs(dataset_root / client_str / alpha_str, exist_ok=True)
    if os.path.isdir(pickles_dir):
        os.system(f"rm -rf {pickles_dir}")
    os.mkdir(pickles_dir)

    client_num_in_total = args.client_num_in_total
    ori_dataset, target_dataset = DATASET[args.dataset]

    if args.clip:
        _, transform = clip.load("ViT-B/32")
        trainset = ori_dataset(dataset_root, train=True, download=True)
        testset = ori_dataset(dataset_root, train=False)
    else:
        trainset = ori_dataset(dataset_root, train=True, download=True)
        testset = ori_dataset(dataset_root, train=False)
    # concat_datasets = [trainset, testset]
    concat_datasets = [trainset]  # NOTE: only trainset

    # Dirichlet(alpha)
    all_datasets, stats = dirichlet_distribution(
        ori_dataset=concat_datasets,
        target_dataset=target_dataset,
        num_clients=client_num_in_total,
        alpha=args.alpha,
        transform=transform,
        target_transform=target_transform,
    )

    for subset_id, client_id in enumerate(
        range(0, len(all_datasets), args.client_num_in_each_pickles)
    ):
        subset = all_datasets[client_id : client_id + args.client_num_in_each_pickles]
        with open(pickles_dir / str(subset_id) + ".pkl", "wb") as f:
            pickle.dump(subset, f)

    # save testdata
    data_numpy = np.concatenate([testset.data], axis=0, dtype=np.float32)
    targets_numpy = np.concatenate([testset.targets], axis=0, dtype=np.int64)
    num_samples = int(len(testset) * args.testset_ratio)
    indices = list(range(len(testset)))
    sampled_indices = random.sample(indices, num_samples)
    sampled_data = data_numpy[sampled_indices]
    sampled_targets = targets_numpy[sampled_indices]

    testset_extracted = target_dataset(
        data=sampled_data,
        targets=sampled_targets,
        transform=transform,
        target_transform=target_transform,
    )

    with open(pickles_dir / f"testdata.pkl", "wb") as f:
        pickle.dump(testset_extracted, f)

    # save stats
    client_id_indices = [i for i in range(client_num_in_total)]
    with open(pickles_dir / f"seperation.pkl", "wb") as f:
        pickle.dump({"id": client_id_indices, "total": client_num_in_total}, f)
    with open(DATASETS_DIR / args.dataset / "all_stats.json", "w") as f:
        json.dump(stats, f)
    print("write stats to", DATASETS_DIR / args.dataset / "all_stats.json")

    args.root = (
        Path(args.root).abspath()
        if str(dataset_root) != str(DATASETS_DIR / args.dataset)
        else None
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--clip", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100"],
        default="cifar10",
    )
    ################# Dirichlet distribution only #################
    parser.add_argument(
        "--alpha",
        type=float,
        default=0,
        help="Only for controling data hetero degree while performing Dirichlet partition.",
    )
    ###############################################################
    parser.add_argument("--client_num_in_total", type=int, default=100)
    parser.add_argument(
        "--classes",
        type=int,
        default=-1,
        help="Num of classes that one client's data belong to.",
    )
    parser.add_argument("--seed", type=int, default=0)
    #######################################################
    parser.add_argument("--client_num_in_each_pickles", type=int, default=10)
    parser.add_argument("--testset_ratio", type=float, default=0.2)
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()

    main(args)
    dataset = args.dataset
    alpha_str = str(args.alpha).replace(".", "_")
    client_str = str(args.client_num_in_total)
    args_dict = dict(args._get_kwargs())
    if args.clip:
        with open(
            DATASETS_DIR / dataset / "clip" / client_str / alpha_str / "args.json", "w"
        ) as f:
            json.dump(args_dict, f)
    else:
        with open(
            DATASETS_DIR / dataset / client_str / alpha_str / "args.json", "w"
        ) as f:
            json.dump(args_dict, f)

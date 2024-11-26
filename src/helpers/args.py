import torch
from argparse import ArgumentParser
from helpers.config import *


def get_args():
    parser = ArgumentParser()
    ##### basic setup #####
    parser.add_argument("--global_epochs", type=int, default=500)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--local_steps", type=int, default=0)
    parser.add_argument("--local_lr", type=float, default=1e-2)
    parser.add_argument("--lr_decay", type=float, default=1)
    parser.add_argument("--lr_decay_step", type=float, default=1)
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--verbose_gap", type=int, default=10)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "domainnet", "xglue"],
        default="cifar10",
    )
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--model_type", type=str, default="clip")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_data_max", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_log", default=False, action="store_true")
    parser.add_argument("--save_period", type=int, default=1)
    parser.add_argument("--save_model", default=False, action="store_true")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--only_test", default=False, action="store_true")
    parser.add_argument("--only_train", default=False, action="store_true")

    ##### FL setup #####
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Only for controling data hetero degree while performing Dirichlet partition.",
    )
    parser.add_argument("--client_num_in_total", type=int, default=100)
    parser.add_argument("--client_num_per_round", type=int, default=20)
    
    ##### layer selection policy #####
    parser.add_argument(
        "--n_layers", type=int, default=1, help="number of layers to train"
    )
    parser.add_argument("--n_layers_inc", type=int, default=0, help="")
    parser.add_argument(
        "--strategy",
        type=str,
        default="pro",
        choices=["top", "bottom", "both", "full", "pro", "rgn", "snr"],
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="diff",
        choices=["diff", "rgn", "snr"],
        help="metric to select layers. diff: model difference (gradient norm); cost: training cost; hybrid: diff and cost",
    )
    parser.add_argument(
        "--sel_gap",
        type=int,
        default=1,
        help="re-select layers every x epochs (-1: only select once at the beginning)",
    )
    parser.add_argument(
        "--sel_batch",
        type=int,
        default=-1,
        help="number of batches for layer selection (default: -1, all batches)",
    )
    parser.add_argument(
        "--fixed_names",
        default=[],
    )
    parser.add_argument(
        "--fixed_layers",
        default=[],
    )
    parser.add_argument(
        "--common_layers",
        default=[],
    )
    parser.add_argument("--balance", type=float, default=0.0, help="balance factor in the proposed method")

    args = parser.parse_args()

    if args.model_type == "clip":
        args.img_dtype = torch.float16
        for key, value in CLIP_CONFIG.items():
            if hasattr(args, key):
                setattr(args, key, value)
        
    elif args.model_type in (
        "xlm-roberta-large",
        "roberta-large",
        "xlm-roberta-base",
        "roberta-base",
    ):
        args.img_dtype = torch.float32
        for key, value in ROBERTA_BASE_CONFIG.items():
            if hasattr(args, key):
                setattr(args, key, value)

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    args.preprocess = None
    return args

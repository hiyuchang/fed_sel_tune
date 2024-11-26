import numpy as np
from collections import Counter


def dirichlet_distribution(
    ori_dataset,
    target_dataset,
    num_clients,
    alpha,
    transform=None,
    target_transform=None,
):
    NUM_CLASS = len(ori_dataset[0].classes)
    MIN_SIZE = 0
    X = [[] for _ in range(num_clients)]
    Y = [[] for _ in range(num_clients)]
    stats = {}
    targets_numpy = np.concatenate(
        [ds.targets for ds in ori_dataset], axis=0, dtype=np.int64
    )
    data_numpy = np.concatenate(
        [ds.data for ds in ori_dataset], axis=0, dtype=np.float32
    )
    idx = [np.where(targets_numpy == i)[0] for i in range(NUM_CLASS)]

    while MIN_SIZE < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(NUM_CLASS):
            np.random.shuffle(idx[k])
            distributions = np.random.dirichlet(np.repeat(alpha, num_clients))
            distributions = np.array(
                [
                    p * (len(idx_j) < len(targets_numpy) / num_clients)
                    for p, idx_j in zip(distributions, idx_batch)
                ]
            )
            distributions = distributions / distributions.sum()
            distributions = (np.cumsum(distributions) * len(idx[k])).astype(int)[:-1]
            idx_batch = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(idx_batch, np.split(idx[k], distributions))
            ]
            MIN_SIZE = min([len(idx_j) for idx_j in idx_batch])

        for i in range(num_clients):
            stats[i] = {"x": None, "y": None}
            np.random.shuffle(idx_batch[i])
            X[i] = data_numpy[idx_batch[i]]
            Y[i] = targets_numpy[idx_batch[i]]
            stats[i]["x"] = len(X[i])
            stats[i]["y"] = Counter(Y[i].tolist())

    datasets = [
        target_dataset(
            data=X[j],
            targets=Y[j],
            transform=transform,
            target_transform=target_transform,
        )
        for j in range(num_clients)
    ]
    return datasets, stats


def dirichlet_dict(dataset, n_parties, alpha, num_classes, ds_name=None):
    min_size = 0
    min_require_size = 10
    if ds_name in ("openbookqa", "arc_easy", "arc_challenge"):
        answers = dataset["answerKey"]
        y_train = np.array([ord(a) - ord("A") for a in answers])
    elif ds_name == "piqa":
        y_train = np.array(dataset["label"])
    elif ds_name == "20news":
        y_train = np.array(dataset["label"])
    else:
        y_train = np.array(dataset.targets, dtype=np.int64)
    N = len(y_train)

    net_dataidx_map = {}
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(num_classes):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
            proportions = np.array(
                [
                    p * (len(idx_j) < N / n_parties)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    # print("net_dataidx_map", {k: len(v) for k, v in net_dataidx_map.items()})
    return net_dataidx_map

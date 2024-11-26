
import torch
from path import Path
MAIN_DIR = Path(__file__).parent.parent.parent.parent.abspath()
DATASETS_DIR = MAIN_DIR / "datasets"
DOMAINNET_SITE = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]


def prepare_domain_data(args, only_size, client_id=None):
    from data_helpers.dataset import DomainNetDataset

    N = args.client_num_in_total
    K = len(DOMAINNET_SITE)
    n = int(N // K)
    num_client_per_type = [n] * K
    if sum(num_client_per_type) != N:
        remain = N - sum(num_client_per_type)
        for i in range(remain):
            num_client_per_type[i] += 1

    # calculate dataset size
    domain_num = []
    for site in DOMAINNET_SITE:
        trainset = DomainNetDataset(DATASETS_DIR, site, train=True, transform=None)
        domain_num.append(len(trainset))
    datasize_dict = {}
    for i in range(N):
        datasize_dict[i] = domain_num[i % K] // num_client_per_type[i % K]
    if only_size:
        return datasize_dict

    site = DOMAINNET_SITE[client_id % K]
    k = client_id % K
    index_start = sum([datasize_dict[x * K + k] for x in range(client_id // K)])

    # trainset
    train_domain = DomainNetDataset(
        DATASETS_DIR, site, train=True, transform=args.preprocess
    )
    trainset = torch.utils.data.Subset(
        train_domain, list(range(index_start, index_start + datasize_dict[client_id]))
    )

    # testset
    tmp_list = []
    for site in DOMAINNET_SITE:
        test_domain = DomainNetDataset(
            DATASETS_DIR, site, train=False, transform=args.preprocess
        )
        tmp_list.append(test_domain)
    testset = torch.utils.data.ConcatDataset(tmp_list)

    return trainset, testset

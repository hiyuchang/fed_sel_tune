import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Grayscale
from PIL import Image
from path import Path
MAIN_DIR = Path(__file__).parent.parent.parent.parent.abspath()
DATASETS_DIR = MAIN_DIR / "datasets"


class CIFARDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        _data = data
        _targets = targets
        if not isinstance(_data, torch.Tensor):
            if not isinstance(_data, np.ndarray):
                _data = ToTensor()(_data)
            else:
                _data = torch.tensor(_data)
        self.data = torch.permute(_data, [0, -1, 1, 2]).float()
        if not isinstance(_targets, torch.Tensor):
            _targets = torch.tensor(_targets)
        self.targets = _targets.long()

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return data, targets

    def __len__(self):
        return len(self.targets)


class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load(
                base_path + "/DomainNet/{}_train.pkl".format(site), allow_pickle=True
            )
        else:
            self.paths, self.text_labels = np.load(
                base_path + "/DomainNet/{}_test.pkl".format(site), allow_pickle=True
            )

        label_dict = {
            "bird": 0,
            "feather": 1,
            "headphones": 2,
            "ice_cream": 3,
            "teapot": 4,
            "tiger": 5,
            "whale": 6,
            "windmill": 7,
            "wine_glass": 8,
            "zebra": 9,
        }

        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else "../datasets"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

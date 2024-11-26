import torch
import torch.nn as nn


class CLIP_CIFAR10(nn.Module):
    def __init__(self, clip_model, num_classes=10):
        super().__init__()
        self.clip = clip_model
        self.classifier = nn.Linear(512, num_classes, dtype=torch.float16)

    def forward(self, x):
        x = self.clip(x)
        out = self.classifier(x)
        return out

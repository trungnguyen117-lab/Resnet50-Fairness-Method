# flac/models/resnet

import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet50


class ResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model=None):
        super().__init__()
        if model is None:
            model = resnet18(pretrained=pretrained)
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
            self.embed_size = 512  # ResNet-18 có output feature size là 512
            self.num_classes = num_classes
            self.fc = nn.Linear(self.embed_size, num_classes)
        else:
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
            self.embed_size = 512
            self.num_classes = num_classes
            self.fc = model.fc
        print(f"ResNet18 - num_classes: {num_classes} pretrained: {pretrained}")

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        logits = self.fc(out)
        feat = F.normalize(out, dim=1)
        return logits, feat


class ResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model=None):
        super().__init__()
        if model is None:
            model = resnet50(pretrained=pretrained)
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
            self.embed_size = 2048  # ResNet-50 có output feature size là 2048
            self.num_classes = num_classes
            self.fc = nn.Linear(self.embed_size, num_classes)
        else:
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
            self.embed_size = 2048
            self.num_classes = num_classes
            self.fc = model.fc
        print(f"ResNet50 - num_classes: {num_classes} pretrained: {pretrained}")

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        logits = self.fc(out)
        feat = F.normalize(out, dim=1)
        return logits, feat

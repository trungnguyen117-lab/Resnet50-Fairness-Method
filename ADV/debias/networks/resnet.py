import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet18, resnet50

class FCResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        model = resnet18(pretrained=pretrained)
        modules = list(model.children())[:-1]
        self.extractor = nn.Sequential(*modules)
        self.embed_size = 512
        print(f'FCResNet18 - num_classes: {num_classes} pretrained: {pretrained}')

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        feat = F.normalize(out, dim=1)
        return feat


class FCResNet18Base(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        model = resnet18(pretrained=pretrained)
        modules = list(model.children())[:-1]
        self.extractor = nn.Sequential(*modules)
        self.embed_size = 512
        print(f'FCResNet18Base - num_classes: {num_classes} pretrained: {pretrained}')

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        feat = F.normalize(out, dim=1)
        return feat


class FCResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, hidden_size=2048, dropout=0.5):
        super().__init__()
        self.resnet = resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Linear(2048, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)
        features = self.dropout(self.relu(features))
        return features


class FCResNet18_Base(nn.Module):
    """ResNet18 without the final fc layer"""
    
    def __init__(self, pretrained=True, hidden_size=512, dropout=0.5):
        super().__init__()
        self.resnet = resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(512, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)
        features = self.dropout(self.relu(features))
        return features


# Sử dụng mô hình
if __name__ == "__main__":
    # Khởi tạo mô hình
    model_resnet18 = FCResNet18(num_classes=2, pretrained=True)
    model_resnet18_base = FCResNet18Base(num_classes=2, pretrained=True)
    model_resnet50 = FCResNet50(num_classes=2, pretrained=True)
    model_resnet18_base_only = FCResNet18_Base(pretrained=True)

    # Tạo một tensor ví dụ
    input_tensor = torch.randn(1, 3, 224, 224)  # Ví dụ với batch size là 1 và kích thước ảnh là 224x224

    # Truyền qua các mô hình
    features_resnet18 = model_resnet18(input_tensor)
    features_resnet18_base = model_resnet18_base(input_tensor)
    features_resnet50 = model_resnet50(input_tensor)
    features_resnet18_base_only = model_resnet18_base_only(input_tensor)

    # In kết quả
    print(f"FCResNet18 features: {features_resnet18.shape}")
    print(f"FCResNet18Base features: {features_resnet18_base.shape}")
    print(f"FCResNet50 features: {features_resnet50.shape}")
    print(f"FCResNet18_Base features: {features_resnet18_base_only.shape}")

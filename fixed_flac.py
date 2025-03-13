import os
import sys
import time
import torch
import logging
import argparse
import datetime
import numpy as np
from torch import nn
from tqdm import tqdm
from pathlib import Path
from numpy import mean, std

project_dir = "/root/DL-Fairness-Study"
sys.path.insert(1, os.path.join(project_dir, "FLAC"))

from flac import flac_loss
from torchvision.models import resnet50
from utils.logging import set_logging
from datasets.utk_face import get_utk_face
from utils.utils import (
    AverageMeter,
    MultiDimAverageMeter,
    accuracy,
    load_model,
    pretty_dict,
    save_model,
    set_seed,
)

sys.path.insert(1, project_dir)

from arguments import get_args
from metrics import get_metric_index, get_all_metrics, print_all_metrics

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet50, self).__init__()
        # Tạo model ResNet50 không pretrained
        self.model = resnet50(pretrained=False)
        # Thay đổi fully connected layer cuối
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def get_features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        return features, self.model.fc(features)

def set_model(opt):
    model = CustomResNet50(num_classes=2)
    model = model.cuda()
    
    criterion1 = nn.CrossEntropyLoss()
    
    # Protected network
    protected_net = CustomResNet50(num_classes=2)
    
    if opt.sensitive == "race":
        protected_attr_model = f"{project_dir}/FLAC/bias_capturing_classifiers/bcc_race.pth"
    elif opt.sensitive == "age":
        protected_attr_model = f"{project_dir}/FLAC/bias_capturing_classifiers/bcc_age.pth"
    
    # Load state dict with error handling
    try:
        state_dict = load_model(protected_attr_model)
        # Xử lý state dict trước khi load
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # remove 'module.' prefix
            if k.startswith('model.'):
                k = k[6:]  # remove 'model.' prefix
            new_state_dict[k] = v
        
        protected_net.model.load_state_dict(new_state_dict, strict=False)
        print("Successfully loaded protected network weights")
    except Exception as e:
        print(f"Error loading protected network weights: {e}")
        print("Initializing protected network with random weights")
    
    protected_net = protected_net.cuda()
    return model, criterion1, protected_net

def train(train_loader, model, criterion, optimizer, protected_net, opt):
    model.train()
    avg_loss = AverageMeter()
    avg_clloss = AverageMeter()
    avg_miloss = AverageMeter()
    total_b_pred = 0
    total = 0
    
    for images, labels, biases, _ in train_loader:
        bsz = labels.shape[0]
        labels, biases = labels.cuda(), biases.cuda()
        images = images.cuda()
        
        # Get features and logits
        features, logits = model.get_features(images)
        
        with torch.no_grad():
            pr_feat, pr_l = protected_net.get_features(images)
            predicted_race = pr_l.argmin(dim=1, keepdim=True)
        
        predicted_race = predicted_race.T
        
        # Calculate losses
        loss_mi_div = opt.alpha * (flac_loss(pr_feat, features, labels))
        loss_cl = 0.01 * criterion(logits, labels)
        loss = loss_cl + loss_mi_div
        
        # Update metrics
        avg_loss.update(loss.item(), bsz)
        avg_clloss.update(loss_cl.item(), bsz)
        avg_miloss.update(loss_mi_div.item(), bsz)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_b_pred += predicted_race.eq(biases.view_as(predicted_race)).sum().item()
        total += bsz
        
    return avg_loss.avg, avg_clloss.avg, avg_miloss.avg

def validate(val_loader, model):
    model.eval()
    top1 = AverageMeter()
    attrwise_acc_meter = MultiDimAverageMeter(dims=(2, 2))
    
    with torch.no_grad():
        for images, labels, biases, _ in val_loader:
            images, labels, biases = images.cuda(), labels.cuda(), biases.cuda()
            bsz = labels.shape[0]
            
            # Get features and predictions
            _, output = model.get_features(images)
            preds = output.data.max(1, keepdim=True)[1].squeeze(1)
            
            # Calculate accuracy
            (acc1,) = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)
            
            # Update attribute-wise accuracy
            corrects = (preds == labels).long()
            attrwise_acc_meter.add(corrects.cpu(), torch.stack([labels.cpu(), biases.cpu()], dim=1))
    
    return top1.avg, attrwise_acc_meter.get_mean()

def main():
    opt = get_args()
    exp_name = f"flac-utkface_{opt.sensitive}-resnet50-lr{opt.lr}-bs{opt.batch_size}-epochs{opt.epochs}-alpha{opt.alpha}-seed{opt.seed}"
    
    output_dir = f"{project_dir}/checkpoints/{exp_name}"
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Set seed: {opt.seed}")
    set_seed(opt.seed)
    print(f"save_path: {save_path}")
    
    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)
    
    # Data loading
    root = f"{project_dir}/data/utkface"
    train_loader = get_utk_face(
        root,
        batch_size=opt.batch_size,
        bias_attr=opt.sensitive,
        split="train",
        aug=False,
    )
    
    val_loaders = {
        "valid": get_utk_face(root, batch_size=256, bias_attr=opt.sensitive, split="valid", aug=False),
        "test": get_utk_face(root, batch_size=256, bias_attr=opt.sensitive, split="test", aug=False)
    }
    
    # Model setup
    model, criterion, protected_net = set_model(opt)
    
    # Load checkpoint if specified
    if opt.checkpoint:
        try:
            checkpoint = torch.load(f"{save_path}/best_model.pt")
            model.load_state_dict(checkpoint["model"])
            print("Successfully loaded checkpoint")
            
            # Evaluate and exit
            model.eval()
            with torch.no_grad():
                all_labels, all_biases, all_preds = [], [], []
                for images, labels, biases, _ in val_loaders["test"]:
                    images = images.cuda()
                    _, output = model.get_features(images)
                    preds = output.data.max(1, keepdim=True)[1].squeeze(1).cpu()
                    
                    all_labels.append(labels)
                    all_biases.append(biases)
                    all_preds.append(preds)
                
                fin_labels = torch.cat(all_labels)
                fin_biases = torch.cat(all_biases)
                fin_preds = torch.cat(all_preds)
                
                ret = get_all_metrics(y_true=fin_labels, y_pred=fin_preds, sensitive_features=fin_biases)
                print_all_metrics(ret=ret)
            
            sys.exit()
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
    
    # Training setup
    decay_epochs = [opt.epochs // 3, opt.epochs * 2 // 3]
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    
    print(f"decay_epochs: {decay_epochs}")
    
    # Training tracking
    best_accs = {"valid": 0, "test": 0}
    best_epochs = {"valid": 0, "test": 0}
    best_stats = {}
    
    # Training loop
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        print(f"[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}")
        
        # Train
        loss, cllossp, milossp = train(train_loader, model, criterion, optimizer, protected_net, opt)
        print(f"[{epoch} / {opt.epochs}] Loss: {loss:.4f}  Loss CE: {cllossp:.4f}  Loss MI: {milossp:.4f}")
        
        scheduler.step()
        
        # Validate
        stats = pretty_dict(epoch=epoch)
        for key, val_loader in val_loaders.items():
            accs, valid_attrwise_accs = validate(val_loader, model)
            
            stats[f"{key}/acc"] = accs.item()
            stats[f"{key}/acc_unbiased"] = torch.mean(valid_attrwise_accs).item() * 100
            eye_tsr = torch.eye(2)
            stats[f"{key}/acc_skew"] = valid_attrwise_accs[eye_tsr == 0.0].mean().item() * 100
            stats[f"{key}/acc_align"] = valid_attrwise_accs[eye_tsr > 0.0].mean().item() * 100
            
            # Update best stats
            if stats[f"{key}/acc_unbiased"] > best_accs[key]:
                best_accs[key] = stats[f"{key}/acc_unbiased"]
                best_epochs[key] = epoch
                best_stats[key] = pretty_dict(**{f"best_{key}_{k}": v for k, v in stats.items()})
                
                # Save best model
                if key == "valid":
                    save_file = save_path / "best_model.pt"
                    save_model(model, optimizer, opt, epoch, save_file)
            
            print(f"[{epoch} / {opt.epochs}] {key} accuracy: {stats[f'{key}/acc_unbiased']:.3f}")
            print(f"Best {key} accuracy: {best_accs[key]:.3f} at epoch {best_epochs[key]}")
    
    # Print training summary
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")
    print(f"Time per epoch: {total_time / opt.epochs:.6f}")

if __name__ == "__main__":
    main()
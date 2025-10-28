#!/usr/bin/env python3
"""
train_resnet_dropout_singular_values.py

Train ResNet-18 style model on CIFAR-100, apply dropout (pre-ReLU in each BasicBlock),
run multiple seeds and dropout rates, collect test accuracy and singular values of activation matrices,
and save results to an Excel file.

"""

import argparse
import os
import random
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

# -------------------------
# Model: BasicBlock with dropout pre-ReLU
# -------------------------
class BasicBlockDropout(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super(BasicBlockDropout, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        # Skip connection for dimension mismatch
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)  # Dropout after activation (stable)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu2(out)
        return out


class ResNetDropout(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, dropout_rate=0.0):
        super(ResNetDropout, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1, dropout_rate=0.0)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout_rate=dropout_rate)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet18_Dropout(dropout_rate=0.0):
    return ResNetDropout(BasicBlockDropout, [2,2,2,2], dropout_rate=dropout_rate)


# -------------------------
# Utilities: seed, singular values
# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def compute_singular_values(feature_tensor, top_k=10):
    """
    feature_tensor: torch.Tensor with shape (batch, C, H, W) or (batch, features)
    We construct A with shape (neurons, batch), i.e. neurons = C*H*W
    Returns top_k singular values as a list.
    """
    if feature_tensor.dim() > 2:
        B = feature_tensor.shape[0]
        A = feature_tensor.view(B, -1).T  # (neurons, batch)
    else:
        B = feature_tensor.shape[0]
        A = feature_tensor.T

    A = A.detach().cpu().float()

    try:
        svals = torch.linalg.svdvals(A)
        # Take top_k singular values, or all if fewer exist
        svals = svals[:min(top_k, svals.numel())].numpy()
    except RuntimeError:
        try:
            U, S, V = torch.svd(A)
            svals = S[:min(top_k, S.numel())].numpy()
        except Exception:
            # Fallback: return zeros if computation fails
            svals = np.zeros(top_k)
    
    # Pad with zeros if fewer than top_k singular values
    if len(svals) < top_k:
        svals = np.pad(svals, (0, top_k - len(svals)), mode='constant')
    
    return svals.tolist()


# -------------------------
# Training / evaluation
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()
    return running_loss / total, correct / total


def eval_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
    return correct / total


# -------------------------
# Main experiment loop
# -------------------------
def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Data transforms for CIFAR-100
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Dataset / loaders
    trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=test_transform)

    # Initialize Excel output
    excel_path = args.out_excel
    os.makedirs(os.path.dirname(excel_path) or ".", exist_ok=True)
    results = []
    top_k = 100  # Number of singular values to store

    layer_names = args.layer_names

    for dr in args.dropout_rates:
        for seed in args.seeds:
            print(f"\n=== Dropout {dr} Seed {seed} ===")
            set_seed(seed)

            trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

            model = ResNet18_Dropout(dropout_rate=dr)
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

            # Register hooks for feature extraction
            features = {}
            def make_hook(name):
                def hook(module, inp, out):
                    features[name] = out.detach().cpu()
                return hook

            for name in layer_names:
                layer = getattr(model, name, None)
                if layer is None:
                    raise ValueError(f"Model has no layer named {name}")
                layer.register_forward_hook(make_hook(name))

            checkpoints = sorted(list(set([1, max(1, args.epochs//2), args.epochs])))
            print("Checkpoints:", checkpoints)

            for epoch in range(1, args.epochs + 1):
                t0 = time.time()
                train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
                scheduler.step()
                test_acc = eval_model(model, testloader, device)
                t1 = time.time()

                if epoch % args.log_every == 0 or epoch in checkpoints:
                    print(f"Dropout {dr} Seed {seed} Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f} time={t1-t0:.1f}s")

                if epoch in checkpoints:
                    model.eval()
                    with torch.no_grad():
                        for inputs, _ in testloader:
                            inputs = inputs.to(device)
                            _ = model(inputs)
                            break

                    for lname in layer_names:
                        if lname not in features:
                            print("Warning: no feature for", lname)
                            continue
                        svals = compute_singular_values(features[lname], top_k=top_k)
                        results.append({
                            'dropout': dr,
                            'seed': seed,
                            'epoch': epoch,
                            'layer': lname,
                            'singular_values': ','.join(f"{x:.6f}" for x in svals),
                            'test_acc': test_acc
                        })

    # Save to Excel
    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)
    print("DONE. Results written to:", excel_path)


# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 on CIFAR100 with pre-ReLU dropout and collect singular values.")
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--dropout_rates', nargs='+', type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # Updated to match provided data
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--milestones', nargs='+', type=int, default=[50, 75])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--out_excel', type=str, default='results_singular_values.xlsx')
    parser.add_argument('--layer_names', nargs='+', type=str, default=['layer1', 'layer2', 'layer4'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
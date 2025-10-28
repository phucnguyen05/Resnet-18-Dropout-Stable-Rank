#!/usr/bin/env python3
"""
train_resnet_dropout_pca.py

Trains a ResNet-18 model on CIFAR-100 with dropout (pre-ReLU in BasicBlocks of layer2, layer3, layer4).
Runs experiments over multiple seeds and dropout rates, collects test accuracy and singular values of
activation matrices for specified layers, and generates separate PCA scatter plots for each dropout rate,
layer, and epoch checkpoint. Saves results to an Excel file.

Note: Run locally on a GPU; this script does not execute in this environment.
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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------------
# Model: ResNet-18 with Dropout
# -------------------------
class BasicBlockDropout(nn.Module):
    """Basic residual block with dropout applied pre-ReLU."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        # Skip connection for dimension mismatch
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)  # Dropout after first ReLU
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu2(out)
        return out

class ResNetDropout(nn.Module):
    """ResNet-18 with dropout in specified layers."""
    def __init__(self, block, num_blocks, num_classes=100, dropout_rate=0.0):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # Dropout applied only in layer2, layer3, layer4
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout_rate=0.0)
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
    """Create ResNet-18 with specified dropout rate."""
    return ResNetDropout(BasicBlockDropout, [2, 2, 2, 2], dropout_rate=dropout_rate)

# -------------------------
# Utilities
# -------------------------
def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def compute_singular_values(feature_tensor, top_k=100):
    """
    Compute top_k singular values of activation matrix.
    
    Args:
        feature_tensor (torch.Tensor): Shape (batch, C, H, W) or (batch, features)
        top_k (int): Number of singular values to return
    
    Returns:
        list: Top k singular values, padded with zeros if needed
    """
    if feature_tensor.dim() > 2:
        B = feature_tensor.shape[0]
        A = feature_tensor.view(B, -1).T  # (neurons, batch)
    else:
        A = feature_tensor.T
    A = A.detach().cpu().float()
    try:
        svals = torch.linalg.svdvals(A)
        svals = svals[:min(top_k, svals.numel())].numpy()
    except RuntimeError:
        try:
            U, S, V = torch.svd(A)
            svals = S[:min(top_k, S.numel())].numpy()
        except Exception:
            svals = np.zeros(top_k)
    if len(svals) < top_k:
        svals = np.pad(svals, (0, top_k - len(svals)), mode='constant')
    return svals.tolist()

def compute_pca_activations(feature_tensor):
    """
    Compute 2D PCA projections of activations.
    
    Args:
        feature_tensor (torch.Tensor): Shape (batch, C, H, W) or (batch, features)
    
    Returns:
        np.ndarray: PCA projections with shape (batch, 2)
    """
    if feature_tensor.dim() > 2:
        A = feature_tensor.view(feature_tensor.shape[0], -1).numpy()  # (batch, neurons)
    else:
        A = feature_tensor.numpy()
    pca = PCA(n_components=2)
    try:
        pca_result = pca.fit_transform(A)  # (batch, 2)
    except Exception:
        pca_result = np.zeros((A.shape[0], 2))
    return pca_result

# -------------------------
# Training and Evaluation
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch and return loss and accuracy."""
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
    """Evaluate model on test set and return accuracy."""
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
# Plotting
# -------------------------
def plot_pca(pca_data, layer, epoch, dropout, output_dir="pca_plots"):
    """
    Generate and save PCA scatter plot for a specific layer, epoch, and dropout rate.
    
    Args:
        pca_data (pd.DataFrame): PCA data with columns ['dropout', 'epoch', 'layer', 'pc1', 'pc2']
        layer (str): Layer name (e.g., 'layer1')
        epoch (int): Epoch number
        dropout (float): Dropout rate
        output_dir (str): Directory to save plots
    """
    plt.figure(figsize=(6, 4))
    subset = pca_data[(pca_data['layer'] == layer) & (pca_data['epoch'] == epoch) & (pca_data['dropout'] == dropout)]
    if len(subset) > 0:
        plt.scatter(subset['pc1'], subset['pc2'], alpha=0.5, s=10, c='b')
        plt.title(f'PCA of Activations: {layer}, Epoch {epoch}, Dropout {dropout}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f'pca_{layer}_dropout{dropout}_epoch{epoch}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"PCA plot saved to: {plot_path}")

# -------------------------
# Main Experiment
# -------------------------
def run_experiment(args):
    """Run experiment: train model, collect singular values and PCA, save results and plots."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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

    # Load datasets
    trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=test_transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Initialize results storage
    results = []
    pca_data = []
    top_k = 100  # Number of singular values to store
    layer_names = args.layer_names
    dropout_rates = args.dropout_rates
    checkpoints = sorted(list(set([1, max(1, args.epochs//2), args.epochs])))

    for dr in dropout_rates:
        for seed in args.seeds:
            print(f"\n=== Dropout {dr} Seed {seed} ===")
            set_seed(seed)

            # Initialize model, optimizer, and scheduler
            model = ResNet18_Dropout(dropout_rate=dr).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

            # Register hooks for activation extraction
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

            print(f"Checkpoints: {checkpoints}")

            # Training loop
            for epoch in range(1, args.epochs + 1):
                t0 = time.time()
                train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
                scheduler.step()
                test_acc = eval_model(model, testloader, device)
                t1 = time.time()

                if epoch % args.log_every == 0 or epoch in checkpoints:
                    print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, time={t1-t0:.1f}s")

                # Collect data at checkpoints
                if epoch in checkpoints:
                    model.eval()
                    with torch.no_grad():
                        for inputs, _ in testloader:
                            inputs = inputs.to(device)
                            _ = model(inputs)  # Populate features via hooks
                            break

                    for lname in layer_names:
                        if lname not in features:
                            print(f"Warning: no feature for {lname}")
                            continue
                        # Compute singular values and PCA
                        svals = compute_singular_values(features[lname], top_k=top_k)
                        pca_result = compute_pca_activations(features[lname])
                        # Store results
                        results.append({
                            'dropout': dr,
                            'seed': seed,
                            'epoch': epoch,
                            'layer': lname,
                            'singular_values': ','.join(f"{x:.6f}" for x in svals),
                            'test_acc': test_acc
                        })
                        # Store PCA data
                        for i in range(pca_result.shape[0]):
                            pca_data.append({
                                'dropout': dr,
                                'seed': seed,
                                'epoch': epoch,
                                'layer': lname,
                                'pc1': pca_result[i, 0],
                                'pc2': pca_result[i, 1]
                            })

    # Save results to Excel
    df = pd.DataFrame(results)
    df.to_excel(args.out_excel, index=False)
    print(f"DONE. Results written to: {args.out_excel}")

    # Generate separate PCA plots
    pca_df = pd.DataFrame(pca_data)
    for layer in layer_names:
        for epoch in checkpoints:
            for dr in dropout_rates:
                plot_pca(pca_df, layer, epoch, dr, output_dir="pca_plots")

# -------------------------
# CLI
# -------------------------
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-100 with dropout, collect singular values, and generate PCA plots.")
    parser.add_argument('--data_dir', default='./data', type=str, help='Directory for CIFAR-100 dataset')
    parser.add_argument('--dropout_rates', nargs='+', type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], help='Dropout rates to test')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0], help='Random seeds for experiments')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loaders')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--milestones', nargs='+', type=int, default=[50, 75], help='Epochs for learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate decay factor')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--log_every', type=int, default=1, help='Log frequency (epochs)')
    parser.add_argument('--out_excel', type=str, default='results_singular_values.xlsx', help='Output Excel file path')
    parser.add_argument('--layer_names', nargs='+', type=str, default=['layer1', 'layer2', 'layer4'], help='Layers to analyze')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
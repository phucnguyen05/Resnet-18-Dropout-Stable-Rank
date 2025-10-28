#!/usr/bin/env python3
"""
quick_fc_dropout_stablerank.py

Quick experiment: Apply dropout before the final FC layer of ResNet-18 on CIFAR-100,
compute stable rank of activations for early (layer1), middle (layer2), and late (layer4) layers,
plot stable rank vs dropout for a single seed.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
BATCH_SIZE = 128
EPOCHS = 100
LR = 0.1
LAYER_NAMES = ['layer1', 'layer2', 'layer4']  # Early, middle, late layers

# -------------------------
# Set seed
# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# -------------------------
# ResNet-18 with dropout before final FC
# -------------------------
class ResNet18FCdropout(nn.Module):
    def __init__(self, dropout_rate=0.0, num_classes=100):
        super().__init__()
        from torchvision.models.resnet import resnet18
        self.backbone = resnet18(weights=None, num_classes=num_classes)
        # Replace FC with dropout + linear
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# -------------------------
# Stable rank function
# -------------------------
def compute_stable_rank(features):
    """
    features: tensor of shape (batch, C, H, W) or (batch, features)
    stable_rank = ||A||_F^2 / ||A||_2^2, where A is (neurons, batch)
    """
    if features.dim() > 2:
        B = features.shape[0]
        A = features.view(B, -1).T  # (neurons, batch)
    else:
        A = features.T
    A = A.detach().cpu().float()
    fro_sq = (A ** 2).sum().item()
    try:
        sigma_max = torch.linalg.svdvals(A)[0].item()
    except Exception:
        sigma_max = torch.norm(A, 2).item()
    spec_sq = sigma_max ** 2
    return fro_sq / (spec_sq + 1e-8)

# -------------------------
# Data
# -------------------------
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# -------------------------
# Main loop: run for each dropout
# -------------------------
stable_ranks = defaultdict(list)  # {layer_name: [(dropout, stable_rank), ...]}

for dr in DROPOUT_RATES:
    print(f"\n=== Dropout {dr} ===")
    model = ResNet18FCdropout(dropout_rate=dr).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    # Register hooks for layer1, layer2, layer4
    features = {}
    def make_hook(name):
        def hook(module, inp, out):
            features[name] = out
        return hook
    for name in LAYER_NAMES:
        layer = getattr(model.backbone, name)
        layer.register_forward_hook(make_hook(name))

    # Quick training
    model.train()
    for epoch in range(EPOCHS):
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Compute stable rank for each layer on a test batch
    model.eval()
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(DEVICE)
            _ = model(inputs)  # Populate features via hooks
            for name in LAYER_NAMES:
                if name in features:
                    sr = compute_stable_rank(features[name])
                    stable_ranks[name].append((dr, sr))
            break  # Only first batch

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(8, 5))
for name in LAYER_NAMES:
    dropouts, srs = zip(*stable_ranks[name])
    plt.plot(dropouts, srs, marker='o', label=name)
plt.xlabel("Dropout rate (before final FC)")
plt.ylabel("Stable rank of activations")
plt.title("Stable Rank vs Dropout Rate for ResNet-18 Layers")
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs("quick_plots", exist_ok=True)
plt.savefig("quick_plots/stable_rank_vs_fc_dropout_layers.png", dpi=300)
plt.show()
print("✅ Plot saved at quick_plots/stable_rank_vs_fc_dropout_layers.png")
# Helper function to extract features
features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.linalg as linalg
import os
from sklearn.manifold import TSNE
import numpy as np

# Create a folder to save the plots
os.makedirs("plots", exist_ok=True)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, dropout_rate=0.0):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, dropout_rate=0.0):
        super().__init__()
        self.inplanes = 64
        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_rate=dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dropout_rate=0.0):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer, dropout_rate))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, norm_layer, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Helper function to extract features
features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
results = {}

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

trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

batch_size = 128
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

for dr in dropout_rates:
    model = ResNet(BasicBlock, [2,2,2,2], 100, dr)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    num_epochs = 100
    train_losses = []
    train_accs = []
    test_accs = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_acc = correct / total
        test_accs.append(test_acc)
        print(f"Dropout {dr} Epoch {epoch+1}: train_loss {train_loss:.3f}, train_acc {train_acc:.3f}, test_acc {test_acc:.3f}")
    results[dr] = {'train_losses': train_losses, 'train_accs': train_accs, 'test_accs': test_accs}

    # Compute singular values of the Jacobian
    model.eval()
    input_image, _ = testset[0]
    input_image = input_image.to(device).unsqueeze(0)
    flat_input = input_image.flatten()
    flat_input.requires_grad = True
    def flat_forward(flat_x):
        x = flat_x.reshape(1, 3, 32, 32)
        y = model(x)
        return y.flatten()
    J = []
    for i in range(100):
        y = flat_forward(flat_input)
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1
        retain = i < 99
        grad = torch.autograd.grad(y, flat_input, grad_outputs=grad_outputs, retain_graph=retain)[0]
        J.append(grad)
    J = torch.stack(J)
    s = linalg.svdvals(J)
    results[dr]['singular_values'] = s.cpu().numpy()

    # Extract features for t-SNE
    model.avgpool.register_forward_hook(get_features('feats'))
    all_features = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            _ = model(inputs)
            all_features.append(features['feats'].cpu().view(inputs.size(0), -1))
            all_labels.append(labels)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_features.numpy())

    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels.numpy(), cmap='tab20', s=10)
    plt.title(f't-SNE of Features for Dropout {dr}')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=testset.classes, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"plots/tsne_dropout_{dr}.png")
    plt.close()


# Plot loss and accuracy curves
for dr in dropout_rates:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(results[dr]['train_losses'], label='Train Loss')
    plt.title(f'Loss Curve for Dropout {dr}')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(results[dr]['train_accs'], label='Train Acc')
    plt.plot(results[dr]['test_accs'], label='Test Acc')
    plt.title(f'Accuracy Curve for Dropout {dr}')
    plt.legend()
    plt.savefig(f"plots/dropout_{dr}_train_test.png")
    plt.close()


# Plot singular values
plt.figure(figsize=(8, 6))
for dr in dropout_rates:
    s = sorted(results[dr]['singular_values'], reverse=True)
    plt.plot(s, label=f'Dropout {dr}')
plt.title('Singular Values of the Jacobian for Different Dropout Rates')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.legend()
plt.savefig("plots/singular_values.png")
plt.close()
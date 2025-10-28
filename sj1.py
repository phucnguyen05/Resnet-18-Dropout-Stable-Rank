import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 3x3 convolution
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlockPreDropout(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super(BasicBlockPreDropout, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout(out)   # ← Dropout BEFORE ReLU
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, dropout_rate=0.0):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout_rate=dropout_rate)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(dropout_rate=0.0):
    return ResNet(BasicBlockPreDropout, [2,2,2,2], dropout_rate=dropout_rate)


def train_and_eval(dropout_rate, epochs=100, device="cuda"):
    # CIFAR-100 loaders
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

    model = ResNet18(dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_accs, test_accs = [], []
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        # train
        model.train()
        correct, total, running_loss = 0, 0, 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_losses.append(running_loss/len(trainloader))
        train_accs.append(100.*correct/total)

        # eval
        model.eval()
        correct, total, running_loss = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_losses.append(running_loss/len(testloader))
        test_accs.append(100.*correct/total)
        scheduler.step()

        print(f"[Dropout={dropout_rate}] Epoch {epoch+1}: TrainAcc={train_accs[-1]:.2f} TestAcc={test_accs[-1]:.2f}")

    return model, (train_losses, train_accs, test_losses, test_accs)

def compute_jacobian_singular_values(model, device="cuda"):
    model.eval()
    x = torch.randn(1, 3, 32, 32, requires_grad=True).to(device)
    y = model(x)

    # Compute Jacobian of outputs wrt inputs
    num_classes = y.size(1)
    jacobian = []
    for i in range(num_classes):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[0, i] = 1
        grads = torch.autograd.grad(y, x, grad_outputs=grad_outputs, retain_graph=True)[0]
        jacobian.append(grads.view(-1).detach().cpu().numpy())
    J = np.stack(jacobian, axis=0)

    # SVD
    U, S, Vh = np.linalg.svd(J, full_matrices=False)
    return S

dropouts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
results = {}
singular_values = {}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Train for each dropout rate
for d in dropouts:
    model, curves = train_and_eval(d, epochs=100, device=device)
    results[d] = curves
    singular_values[d] = compute_jacobian_singular_values(model, device)

# Plot accuracy curves
plt.figure(figsize=(10,6))
for d in dropouts:
    _, train_acc, _, test_acc = results[d]
    plt.plot(test_acc, label=f"Dropout={d}")
plt.title("Test Accuracy vs Epochs for Different Dropouts")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

# Plot Jacobian singular values
plt.figure(figsize=(10,6))
for d in dropouts:
    svals = singular_values[d]
    plt.plot(range(1, len(svals)+1), svals, label=f"Dropout={d}")
plt.title("Jacobian Singular Value Spectrum")
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.legend()
plt.yscale("log")
plt.show()


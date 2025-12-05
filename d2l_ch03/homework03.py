import torch
from torch import nn, no_grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 准备数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Flatten()])
dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# 定义模型
net = nn.Sequential(nn.Linear(784, 10))

# ----------------- Review Area Start -----------------
# Bug 1 可能在这里
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    for X, y in dataloader:
        # Bug 2 可能在这里
        optimizer.zero_grad()
        output = net(X)
        l = loss_fn(output, y)
        l.backward()

        optimizer.step()

    print(f"Epoch {epoch} finished")
# ----------------- Review Area End -----------------

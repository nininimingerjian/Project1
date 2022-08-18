import numpy as np
import torch
from torch import nn, optim
import torchvision
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_dataset = datasets.MNIST(root='./data/06_MNIST', train=True, transform=transforms.ToTensor,
                               download=True)
test_dataset = datasets.MNIST(root='./data/06_MNIST', train=False, transform=transforms.ToTensor,
                              download=True)

batch_size = 64

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                         shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n1 = nn.Linear(784, 10)  # 28*28,10个分类
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 【64，1，28，28】 -》 【64，784】
        x = x.view(x.size()[0], -1)
        x = self.n1(x)
        x = self.softmax(x)
        return x


class SGD(Optimizer):
    def __init__(self):
        pass

    def step(self):
        pass


def sgd(params, lr):
    with torch.no_grad():
        for p in params:
            p -= p.grad * lr
            p.grad_zero_()


model = Net()
loss = nn.MSELoss
lr = 0.5
optimizer = sgd(model.parameters(), lr=lr)


def train_model():
    for i, data in enumerate(train_loader):
        inputs, labels = data

        out = model(inputs)
        labels = labels.reshape(-1, 1)
        one_hot = torch.zeros(inputs.shape[0], 10).scatter(1, labels, 1)

        l = loss(out, one_hot)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()


def test_model():
    correct = 0
    for i, data in enumerate(test_loader):
        inputs, label = data
        out = model(inputs)
        _, predicted = torch.max(out, 1)
        correct += (predicted == label).sum()

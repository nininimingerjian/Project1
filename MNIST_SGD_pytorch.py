import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim, Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pylab as pl


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n1 = nn.Linear(784, 10)  # 28*28,10个分类

    def forward(self, x):
        # 【64，1，28，28】 -》 【64，784】
        x = torch.flatten(x)
        x = self.n1(x)
        return x


class MyDataSet:
    def __init__(self, dataset, batch_size):
        self.data = dataset
        self.num_samples = dataset.data.shape[0]
        self.bat_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        bat_indices = torch.randperm(self.num_samples)[:self.bat_size].tolist()
        return self.data[bat_indices], bat_indices


model = Net()
lr = 0.01
batch_size = 64
iterations = 1000
loss = nn.CrossEntropyLoss()


def test_model():
    correct = 0

    for i_test, data_test in enumerate(test_loader):
        inputs_test, label = data_test
        out_test = model(inputs_test)
        _, predicted = torch.max(out_test, 1)
        correct = correct + (predicted == label).sum()
    print('Test acc:{0}'.format(correct / len(test_dataset)))


if __name__ == '__main__':
    train_dataset = datasets.MNIST(root='./data/MNIST', train=True, transform=transforms.ToTensor(),
                                   download=True)
    test_dataset = datasets.MNIST(root='./data/MNIST', train=False, transform=transforms.ToTensor(),
                                  download=True)

    n_samples = train_dataset.data.shape[0]
    k = np.random.randint(0, n_samples, n_samples)
    train_loader = DataLoader(dataset=train_dataset, batch_size=n_samples,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=n_samples,
                             shuffle=True)

    # features, labels = next(iter(train_loader))
    # features = torch.tensor(features.view(features.size(0), -1))
    # labels = torch.tensor(labels)
    # out_ = model(features)
    # loss_ = loss(out_, labels)
    # loss_.backward()
    param_mean = copy.deepcopy(model)

    pre_grad = [param_mean for i in range(n_samples)]
    loss_list = []
    for _, data in enumerate(train_loader):
        img = data[0]
        labels = data[1]
        for i in range(n_samples):
            features = img[i]
            features = Variable(features.view(features.size(0), -1))
            label_i = labels[i]
            label_i = torch.tensor([label_i])
            out_ = model(features)

            loss_ = loss(out_, label_i)
            loss_.backward()
            for p in param_mean.parameters():
                p.data.sub_(p.data)
            for p1, p2, p3 in zip(model.parameters(), pre_grad[i].parameters(), param_mean.parameters()):
                p2.data = p1.grad.data
                p3.data += (1.0 / n_samples) * p1.grad.data

        # train_x, train_y = next(iter(train_loader))
        # img = torchvision.utils.make_grid(train_x, nrow=10)
        # img = img.numpy().transpose(1, 2, 0)
        # plt.imshow(img)
        # plt.show()

        loss_pre = 3
        for i in range(20000):
            j = np.random.randint(0, n_samples)

            features = img[j]
            features = Variable(features.view(features.size(0), -1))
            label_j = labels[j]
            label_j = torch.tensor([label_j])
            out_ = model(features)

            # labels = labels.reshape(-1, 1)
            # one_hot = torch.zeros(inputs.shape[0], 10).scatter(1, labels, 1)
            loss_ = loss(out_, label_j)
            if i < 2000:
                if loss_.data < 4:
                    loss_list.append(loss_.data)
            else:
                if loss_.data < 0.1:
                    loss_list.append(loss_.data)

            loss_.backward()

            for param, param_m, pre_gr in zip(model.parameters(), param_mean.parameters(),
                                              pre_grad[j].parameters()):
                update = (param.grad.data - pre_gr.data) + param_m.data
                param_m.data += (param.grad.data - pre_gr.data) * (1.0 / n_samples)
                pre_gr.data = param.grad.data
                param.data.sub_(lr * update)

            if i % 1000 == 0:
                print('i:', i)
                test_model()

    plt.plot(loss_list, 'g-', scalex=[0, 1])
    plt.show()

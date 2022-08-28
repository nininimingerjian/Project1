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
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 【64，1，28，28】 -》 【64，784】
        x = x.view(x.size()[0], -1)
        x = self.n1(x)
        # x = self.softmax(x)
        return x


class MyDataSet():
    def __init__(self, dataset, i_iter):
        self.data = dataset
        self.i_iter = torch.LongTensor(i_iter)

    def __iter__(self):
        return iter(self.i_iter)

    def __len__(self):
        return len(self.i_iter)


model = Net()
# loss = nn.CrossEntropyLoss(reduction='none')
lr = 0.01
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
    batch_size = 64
    k = np.random.randint(0, n_samples, n_samples)
    sampler = MyDataSet(train_dataset, k)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler,
                              shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=True)

    features, labels = next(iter(train_loader))
    features = Variable(features.view(features.size(0), -1))
    labels = Variable(labels)
    out_ = model(features)
    loss_ = loss(out_, labels)
    loss_.backward()
    pre_grad = []
    param_mean = copy.deepcopy(model)
    for p1, p2 in zip(model.parameters(), param_mean.parameters()):
        p2.data = p1.grad.data
    pre_grad = [param_mean for i in range(n_samples)]

    # train_x, train_y = next(iter(train_loader))
    # img = torchvision.utils.make_grid(train_x, nrow=10)
    # img = img.numpy().transpose(1, 2, 0)
    # plt.imshow(img)
    # plt.show()

    loss_list = []

    for i_train, data in enumerate(train_loader):
        inputs, labels = data
        img = Variable(inputs.view(inputs.size(0), -1))
        labels = Variable(labels)

        out = model(img)
        # labels = labels.reshape(-1, 1)
        # one_hot = torch.zeros(inputs.shape[0], 10).scatter(1, labels, 1)

        loss1 = loss(out, labels)
        loss_list.append(loss1.data)

        loss1.backward()

        for param, param_m, pre_gr in zip(model.parameters(), param_mean.parameters(),
                                          pre_grad[k[i_train+1]].parameters()):
            update = (param.grad.data - pre_gr.data) + param_m.data
            param_m.data += (param.grad.data - pre_gr.data) * (1.0 / n_samples)
            pre_gr.data = param.grad.data
            param.data.sub_(lr * update)

        if i_train % 100 == 0:
            print('i:', i_train)
            test_model()

    plt.plot(loss_list, 'g-')
    plt.show()

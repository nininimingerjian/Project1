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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 【64，1，28，28】 -》 【64，784】
        x = x.view(x.size()[0], -1)
        x = self.n1(x)
        x = self.softmax(x)
        return x


model = Net()
# loss = nn.CrossEntropyLoss(reduction='none')
lr = 0.3
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
    train_dataset = datasets.MNIST(root='./data/06_MNIST', train=True, transform=transforms.ToTensor(),
                                   download=True)
    test_dataset = datasets.MNIST(root='./data/06_MNIST', train=False, transform=transforms.ToTensor(),
                                  download=True)

    batch_size = 64

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=True)

    # train_x, train_y = next(iter(train_loader))
    # img = torchvision.utils.make_grid(train_x, nrow=10)
    # img = img.numpy().transpose(1, 2, 0)
    # plt.imshow(img)
    # plt.show()

    num_epochs = 1
    for epoch in range(num_epochs):
        print(f'epoch:{epoch}')
        loss_list = []
        j = []
        for i_train, data in enumerate(train_loader):
            grad = 0
            j.append(i_train)
            inputs, labels = data
            img = Variable(inputs.view(inputs.size(0), -1))
            labels = Variable(labels)

            out = model(img)
            # labels = labels.reshape(-1, 1)
            # one_hot = torch.zeros(inputs.shape[0], 10).scatter(1, labels, 1)

            loss1 = loss(out, labels)
            loss_list.append(loss1.data)

            loss1.backward()
            grad_list = []

            for param in model.parameters():  # 784*10+10=7850
                grad_list.append(param.grad)

            for i, param in enumerate(model.parameters()):
                param.data -= grad_list[i] * lr

            if i_train % 20 == 0:
                test_model()

        plt.plot(loss_list, 'g-')
        plt.show()

import numpy as np
from torch.optim import _functional as F
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torchvision
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
    def __init__(self, params, lr=0.0, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            F.sgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay,
                  momentum,
                  lr,
                  dampening,
                  nesterov)

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

            return loss


# def sgd(params, lr):
#     with torch.no_grad():
#         for p in params:
#             p -= p.grad * lr
#             p.grad_zero_()


model = Net()
loss = nn.MSELoss()
lr = 0.5

# sgd = SGD(model.parameters(), lr=lr)

# optimizer = torch.optim.SGD(model.parameters(), lr=lr)


optimizer = SGD(params=model.parameters(), lr=lr)


def train_model():
    for i, data in enumerate(train_loader):
        inputs, labels = data

        out = model(inputs)
        labels = labels.reshape(-1, 1)
        one_hot = torch.zeros(inputs.shape[0], 10).scatter(1, labels, 1)

        loss1 = loss(out, one_hot)

        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()


def test_model():
    correct = 0
    for i, data in enumerate(test_loader):
        inputs, label = data
        out = model(inputs)
        _, predicted = torch.max(out, 1)
        correct = correct + (predicted == label).sum()
        print('Test acc:{0}'.format(correct / len(test_dataset)))


num_epochs = 1
for epoch in range(num_epochs):
    print(f'epoch:{epoch}')
    train_model()
    test_model()

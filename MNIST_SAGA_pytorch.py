import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim, Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n1 = nn.Linear(784, 10)  # 28*28,10个分类

    def forward(self, x):
        # 【64，1，28，28】 -》 【64，784】
        x = x.view(x.size()[0], -1)
        x = self.n1(x)
        return x


model = Net()
lr = 0.01
batch_size = 64
iterations = 5000
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
    train_loader = DataLoader(dataset=train_dataset, batch_size=n_samples,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=True)

    param_mean = copy.deepcopy(model)

    loss_list = []
    torch.set_printoptions(32)
    for p in param_mean.parameters():
        p.data.sub_(p.data)
    pre_grad = [param_mean for _ in range(n_samples)]
    for _, data in enumerate(train_loader):
        img = data[0]
        labels = data[1]
        for i in range(n_samples):
            features = img[i]
            features = torch.as_tensor(features.view(features.size(0), -1))
            label_i = labels[i]
            label_i = torch.tensor([label_i])
            model1 = copy.deepcopy(model)
            out_ = model1(features)

            loss_ = loss(out_, label_i)

            loss_.backward()

            for p1, p2, p3 in zip(model1.parameters(), pre_grad[i].parameters(), param_mean.parameters()):
                p2.data = p1.grad.data
                p3.data += (1.0 / n_samples) * p1.grad.data

        for i in range(iterations):
            j = np.random.randint(0, n_samples)
            features = img[j]
            features = Variable(features.view(features.size(0), -1))
            label_j = labels[j]
            label_j = torch.tensor([label_j])
            out_ = model(features)
            loss_ = loss(out_, label_j)

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
    plt.axis([0, iterations, 0.000001, 0.1])
    plt.scatter(x=[l for l in range(len(loss_list))], y=loss_list, s=10)
    # plt.plot(loss_list,'g-')
    plt.show()

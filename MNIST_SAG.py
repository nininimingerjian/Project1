import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
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
iterations = 10000
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
                              shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=True)

    param_d = copy.deepcopy(model)
    for p in param_d.parameters():
        p.data.sub_(p.data)
    param_y_list = [param_d for _ in range(n_samples)]
    torch.set_printoptions(precision=15)
    loss_list = []
    for _, data in enumerate(train_loader):
        img = data[0]
        labels = data[1]

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
            for p, d, y in zip(model.parameters(), param_d.parameters(), param_y_list[j].parameters()):
                d.data = d.data - y.data + p.grad.data
                y.data = p.grad.data
                p.data -= lr * d.data / n_samples

            if i % 2000 == 0:
                print('i:', i)
                test_model()
    plt.axis([0, 10000, 0, 1])
    plt.scatter(y=loss_list, x=[l for l in range(len(loss_list))], s=1)
    plt.show()

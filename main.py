import copy
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

batch_size = 64
loss = nn.CrossEntropyLoss()


def smooth_loss(loss_list_):
    last1 = loss_list_[0]
    loss1 = loss_list_
    weight = 0.9
    for i in range(len(loss1)):
        smooth = last1 * weight + (1 - weight) * loss_list_[i]
        loss1[i] = smooth
        last1 = smooth
    return loss1


class MySampler(Sampler):
    def __init__(self, dataset, batchsize):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batchsize  # 每一批数据量
        self.indices = range(len(dataset))  # 生成数据集的索引
        self.count = int(len(dataset) / self.batch_size)  # 一共有多少批

    def __iter__(self):
        for i in range(self.count):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return self.count


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n1 = nn.Linear(784, 10)  # 28*28,10个分类

    def forward(self, x):
        # 【64，1，28，28】 -》 【64，784】
        x = x.view(x.size()[0], -1)
        x = self.n1(x)
        return x


def init_model_param(model):
    for p in model.parameters():
        p.data.sub_(p.data)


model_origin = Net()


def SVRG_train():
    model = copy.deepcopy(model_origin)
    init_model_param(model)
    lr = 0.18
    m = 50
    iterations = 40
    w0 = Net()
    w_s = copy.deepcopy(w0)
    loss_list = []
    for iter_ in range(iterations):
        w = w_s
        u = copy.deepcopy(w)
        for p in u.parameters():
            p.data.sub_(p.data)
        inputs_, labels_ = torch.as_tensor(inputs), torch.as_tensor(labels)
        out_ = w(inputs_.to(torch.float32))
        loss_u = loss(out_, labels_)
        loss_u.backward()
        for p1, p2 in zip(w.parameters(), u.parameters()):
            p2.data = p1.grad.data * (1.0 / n_samples)
        w0 = w

        for t in range(m):

            i_t = np.random.randint(0, len(data_list))
            features, label_i = data_list[i_t]
            features = torch.as_tensor(features.view(features.size(0), -1))
            label_i = torch.as_tensor(label_i)

            out_m = w0(features.to(torch.float32))
            loss_m = loss(out_m, label_i)
            loss_list.append(loss_m.data)
            loss_m.backward()

            out_w = w(features.to(torch.float32))
            loss_w = loss(out_w, label_i)
            loss_w.backward()
            for p1, p2, p3 in zip(w0.parameters(), u.parameters(), w.parameters()):
                update = p1.grad.data - p3.grad.data + p2.data
                p1.data -= lr * update

        for p1, p2 in zip(w_s.parameters(), w0.parameters()):
            p1.data = p2.data
    return smooth_loss(smooth_loss(loss_list))


def SAGA_train():
    model = copy.deepcopy(model_origin)
    init_model_param(model)
    lr = 0.00001
    iterations = 2000
    loss_list = []
    param_mean = copy.deepcopy(model)
    for p in param_mean.parameters():
        p.data.sub_(p.data)
    pre_grad = [param_mean for _ in range(n_samples)]

    for i in range(len(data_list)):
        features, label_i = data_list[i]
        features = torch.as_tensor(features.view(features.size(0), -1))
        label_i = torch.as_tensor(label_i)
        model1 = copy.deepcopy(model)
        out_ = model1(features.to(torch.float32))
        loss_ = loss(out_, label_i)
        loss_.backward()

        for p1, p2, p3 in zip(model1.parameters(), pre_grad[i].parameters(), param_mean.parameters()):
            p2.data = p1.grad.data
            p3.data += (1.0 / n_samples) * p1.grad.data

    for i in range(iterations + 1):
        j = np.random.randint(0, len(data_list))
        features, label_j = data_list[j]
        features = torch.as_tensor(features.view(features.size(0), -1))
        label_j = torch.as_tensor(label_j)

        out_ = model(features.to(torch.float32))
        loss_ = loss(out_, label_j)
        loss_list.append(loss_.data)
        loss_.backward()

        for param, param_m, pre_gr in zip(model.parameters(), param_mean.parameters(),
                                          pre_grad[j].parameters()):
            update = (param.grad.data - pre_gr.data) + param_m.data
            param_m.data += (param.grad.data - pre_gr.data) * (1.0 / n_samples)
            pre_gr.data = param.grad.data
            param.data.sub_(lr * update)
    return smooth_loss(smooth_loss(loss_list))


def SAG_train():
    model = copy.deepcopy(model_origin)
    init_model_param(model)
    iterations = 2000
    lr = 0.8
    param_d = copy.deepcopy(model)
    for p in param_d.parameters():
        p.data.sub_(p.data)
    param_y_list = [param_d for _ in range(len(data_list))]

    loss_list = []

    for i in range(iterations + 1):
        j = np.random.randint(0, len(data_list))
        features, label_j = data_list[j]
        features = torch.as_tensor(features.view(features.size(0), -1))
        label_j = torch.as_tensor(label_j)

        out_ = model(features.to(torch.float32))
        loss_ = loss(out_, label_j)
        loss_.backward()
        for p, d, y in zip(model.parameters(), param_d.parameters(), param_y_list[j].parameters()):
            d.data = d.data - y.data + p.grad.data
            y.data = p.grad.data
            p.data -= lr * d.data * (1.0 / n_samples)

        loss_list.append(loss_.data)
    return smooth_loss(smooth_loss(loss_list))


def Finito_train():
    model = copy.deepcopy(model_origin)
    init_model_param(model)
    lr = 0.1
    iterations = 2000
    grad_mean = copy.deepcopy(model)
    for p in grad_mean.parameters():
        p.data.sub_(p.data)
    grad_pre_list = [grad_mean for _ in range(len(data_list))]
    param_list = [copy.deepcopy(model) for _ in range(len(data_list))]
    param_mean = copy.deepcopy(model)

    loss_list = []

    for i in range(len(data_list)):
        inputs_i, label_i = data_list[i]
        model1 = copy.deepcopy(model)
        inputs_i = torch.as_tensor(inputs_i.view(inputs_i.size(0), -1))
        label_i = torch.as_tensor(label_i)
        out_i = model1(inputs_i.to(torch.float32))
        loss_i = loss(out_i, label_i)
        loss_i.backward()
        for p1, p2, p3 in zip(model1.parameters(), grad_pre_list[i].parameters(), grad_mean.parameters()):
            p2.data = p1.grad.data
            p3.data += p1.grad.data * (1.0 / n_samples)

    for i in range(iterations):
        for p1, p2, p3 in zip(model.parameters(), param_mean.parameters(), grad_mean.parameters()):
            p1.data = p2.data - p3.data * lr

        j = np.random.randint(0, len(data_list))
        for p1, p2, p3 in zip(model.parameters(), param_list[j].parameters(), param_mean.parameters()):
            p3.data += (p1.data - p2.data) * (1.0 / n_samples)
            p2.data = p1.data
        inputs_j, label_j = data_list[j]
        inputs_j = torch.as_tensor(inputs_j.view(inputs_j.size(0), -1))
        label_j = torch.as_tensor(label_j)
        out_i = model(inputs_j.to(torch.float32))
        loss_j = loss(out_i, label_j)
        loss_j.backward()

        loss_list.append(loss_j.data)

        for p1, p2, p3 in zip(model.parameters(), grad_pre_list[j].parameters(), grad_mean.parameters()):
            p3.data += (p1.grad.data - p2.data) * (1.0 / n_samples)
            p2.data = p1.grad.data
    return smooth_loss(smooth_loss(loss_list))


def Finito_Perm_train():
    model = copy.deepcopy(model_origin)
    init_model_param(model)
    lr = 0.1
    grad_mean = copy.deepcopy(model)
    for p in grad_mean.parameters():
        p.data.sub_(p.data)
    grad_pre_list = [grad_mean for _ in range(len(data_list))]
    param_list = [copy.deepcopy(model) for _ in range(len(data_list))]
    param_mean = copy.deepcopy(model)
    for i in range(len(data_list)):
        inputs_i, label_i = data_list[i]
        model1 = copy.deepcopy(model)
        inputs_i = torch.as_tensor(inputs_i.view(inputs_i.size(0), -1))
        label_i = torch.as_tensor(label_i)
        out_i = model1(inputs_i.to(torch.float32))
        loss_i = loss(out_i, label_i)
        loss_i.backward()
        for p1, p2, p3 in zip(model1.parameters(), grad_pre_list[i].parameters(), grad_mean.parameters()):
            p2.data = p1.grad.data
            p3.data += p1.grad.data * (1.0 / n_samples)
    loss_list = []
    for i in range(2):
        j = [a for a in range(len(data_list))]
        random.shuffle(j)
        for k in range(len(data_list)):
            for p1, p2, p3 in zip(model.parameters(), param_mean.parameters(), grad_mean.parameters()):
                p1.data = p2.data - p3.data * lr

            for p1, p2, p3 in zip(model.parameters(), param_list[j[k]].parameters(), param_mean.parameters()):
                p3.data += (p1.data - p2.data) * (1.0 / n_samples)
                p2.data = p1.data
            inputs_j, label_j = data_list[j[k]]
            inputs_j = torch.as_tensor(inputs_j.view(inputs_j.size(0), -1))
            label_j = torch.as_tensor(label_j)
            out_i = model(inputs_j.to(torch.float32))
            loss_j = loss(out_i, label_j)
            loss_j.backward()

            loss_list.append(loss_j.data)
            for p1, p2, p3 in zip(model.parameters(), grad_pre_list[j[k]].parameters(), grad_mean.parameters()):
                p3.data += (p1.grad.data - p2.data) * (1.0 / n_samples)
                p2.data = p1.grad.data
    return smooth_loss(smooth_loss(loss_list))


# def test_model():
#     correct = 0
#
#     for i_test, data_test in enumerate(test_loader):
#         inputs_test, label = data_test
#         out_test = model(inputs_test)
#         _, predicted = torch.max(out_test, 1)
#         correct = correct + (predicted == label).sum()
#     print('Test acc:{0}'.format(correct / len(test_dataset)))


if __name__ == '__main__':
    train_dataset = datasets.MNIST(root='./data/MNIST', train=True, transform=transforms.ToTensor(),
                                   download=True)
    test_dataset = datasets.MNIST(root='./data/MNIST', train=False, transform=transforms.ToTensor(),
                                  download=True)

    n_samples = train_dataset.data.shape[0]
    train_loader = DataLoader(dataset=train_dataset, batch_size=64,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=True)
    my_sampler = MySampler(train_dataset, 64)
    data_loader = DataLoader(train_dataset, batch_sampler=my_sampler)
    data_list = []
    for data in data_loader:
        data_list.append(data)
    inputs, labels = train_dataset.data, train_dataset.targets

    loss_saga = SAGA_train()
    loss_sag = SAG_train()
    loss_svrg = SVRG_train()
    loss_f = Finito_train()
    loss_f_P = Finito_Perm_train()

    plt.plot(loss_saga, 'g-', label='saga')
    plt.plot(loss_sag, 'y-', label='sag')
    plt.plot(loss_svrg, 'b-', label='svrg')
    plt.plot(loss_f, 'r-', label='Finito')
    plt.plot(loss_f_P, 'c-', label='Finito_Perm')
    plt.legend()
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.grid(visible=True, axis='y', linestyle='--')
    plt.show()

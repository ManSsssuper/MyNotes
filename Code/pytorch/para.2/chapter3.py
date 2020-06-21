# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:07:54 2020

@author: ManSsssuper
"""
import torch 
from time import time
print(torch.__version__)

#矢量运算和单次运算比较
a=torch.ones(1000)
b=torch.ones(1000)
start=time()
c=torch.zeros(1000)
for i in range(1000):
    c[i]=a[i]+b[i]
print(time()-start)

start=time()
d=a+b
print(time()-start)

#################线性回归手动实现################
import torch
from matplotlib import pyplot as plt
import numpy as np
import dipt
print(torch.__version__)

#生成数据集
num_inputs=2
num_examples=1000
true_w=[2,-3.4]
true_b=4.2
features=torch.randn(num_examples,num_inputs,dtype=torch.float32)
labels=true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels+=torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32)

dipt.set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)

#读取数据
batch_size = 10
for X, y in dipt.data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
#模型参数
w=torch.tensor(np.random.normal(0,0.01,(num_inputs,1)),dtype=torch.float32)
b=torch.zeros(1,dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

#定义模型
def linreg(X,w,b):
    return torch.mm(X,w)+b
#定义损失函数
def squared_loss(y_hat,y):
    return (y_hat-y.view(y_hat.size()))**2/2

#定义优化算法
def sgd(params,lr,batch_size):
    for param in params:
        param.data-=lr*param.grad/batch_size

#模型训练
lr=0.03
num_epochs=3
net=linreg
loss=squared_loss
for epoch in range(num_epochs):
    for X,y in dipt.data_iter(batch_size,features,labels):
        l=loss(net(X,w,b),y).sum()
        l.backward()
        sgd([w,b],lr,batch_size)
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l=loss(net(features,w,b),labels)
    print('epoch %d,loss %f'%(epoch+1,train_l.mean().item()))

print(true_w,true_b)
print(w,b)

##############################lr的简洁实现################################
import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
torch.manual_seed(1)
print(torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')
#生成数据
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

#读取数据
batch_size=10
dataset=Data.TensorDataset(features,labels)
data_iter=Data.DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True)
for X,y in data_iter:
    print(X,'\n',y)
    break
#定义模型
class LinearNet(nn.Module):
    def __init__(self,n_feature):
        super(LinearNet,self).__init__()
        self.linear=nn.Linear(n_feature,1,bias=True)
    def forward(self,X):
        y=self.linear(X)
        return y
net=LinearNet(num_inputs)
print(net)
#net=nn.Sequential(nn.Linear(num_inputs,1))
#net=nn.Sequential()
#net.add_module('linear',nn.Linear(num_inputs,1))
#from collections import OrderDict
#net=nn.Sequential(OrderDict(['linear',nn.Linear(num_inputs,1)]))
for param in net.parameters():
    print(param)

#初始化模型参数
from torch.nn import init
init.normal_(net.linear.weight,mean=0.0,std=0.01)
init.constant_(net.linear.bias,val=0.0)
for param in net.parameters():
    print(param)

#损失函数
loss=nn.MSELoss()
#优化算法
import torch.optim as optim
optimizer=optim.SGD(net.parameters(),lr=0.03)
print(optimizer)
# 为不同子网络设置不同的学习率
# optimizer =optim.SGD([
#                 # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#                 {'params': net.subnet1.parameters()}, # lr=0.03
#                 {'params': net.subnet2.parameters(), 'lr': 0.01}
#             ], lr=0.03)
# # 调整学习率
# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1 # 学习率为之前的0.1倍
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
dense = net.linear
print(true_w, dense.weight.data)
print(true_b, dense.bias.data)

######################softmax从0实现############################################
import torch
import torchvision
import numpy as np
import dipt

#读取数据
print(torch.__version__)
print(torchvision.__version__)
batch_size = 256
train_iter, test_iter = dipt.load_data_fashion_mnist(batch_size)

#初始化参数
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))
print(X.sum(dim=1, keepdim=True))

#softmax运算
def softmax(X):#x为n*d，n为样本数，d为标签数
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
#模型
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
print(y_hat.gather(1, y.view(-1, 1)))#gather的用法
#交叉熵
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))
#准确度
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()
# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n
num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                dipt.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到
            
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

#预测
X, y = iter(test_iter).next()

true_labels = dipt.get_fashion_mnist_labels(y.numpy())
pred_labels = dipt.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

dipt.show_fashion_mnist(X[0:9], titles[0:9])

#################################softmax简洁实现##################################
import torch
from torch import nn
from torch.nn import init
import numpy as np
import dipt

print(torch.__version__)
batch_size = 256
train_iter, test_iter = dipt.load_data_fashion_mnist(batch_size)
num_outputs = 10

# class LinearNet(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(LinearNet, self).__init__()
#         self.linear = nn.Linear(num_inputs, num_outputs)
#     def forward(self, x): # x shape: (batch, 1, 28, 28)
#         y = self.linear(x.view(x.shape[0], -1))
#         return y
    
# net = LinearNet(num_inputs, num_outputs)

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

from collections import OrderedDict
net = nn.Sequential(
        # FlattenLayer(),
        # nn.Linear(num_inputs, num_outputs)
        OrderedDict([
          ('flatten', FlattenLayer()),
          ('linear', nn.Linear(num_inputs, num_outputs))])
        )
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
loss = nn.CrossEntropyLoss()    
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 5
dipt.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

###########################多层感知机实现################################
import torch
import numpy as np
import dipt
print(torch.__version__)
batch_size = 256
train_iter, test_iter = dipt.load_data_fashion_mnist(batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2
loss = torch.nn.CrossEntropyLoss()
num_epochs, lr = 5, 100.0
dipt.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)'

###############################简洁实现#####################################
import torch
from torch import nn
from torch.nn import init
import numpy as np
import dipt

print(torch.__version__)
num_inputs, num_outputs, num_hiddens = 784, 10, 256
    
net = nn.Sequential(
        dipt.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs), 
        )
    
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)
    batch_size = 256
train_iter, test_iter = dipt.load_data_fashion_mnist(batch_size)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
dipt.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

###############################激活函数测试########################################
import torch
import numpy as np
import matplotlib.pylab as plt
import sys
import dipt
def xyplot(x_vals, y_vals, name):
    dipt.set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'relu')

y.sum().backward()
xyplot(x, x.grad, 'grad of relu')
y=x.sigmoid()
xyplot(x,y,'sigmoid')
x.grad.zero_()
y.sum().backward()
xyplot(x,x.grad,'grad of sigmoid')
y = x.tanh()
xyplot(x, y, 'tanh')
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')

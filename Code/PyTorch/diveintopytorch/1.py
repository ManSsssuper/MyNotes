# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:50:22 2020

@author: hp
"""
#import torch
#x=torch.empty(5,3)
#print(x)
#x=torch.rand(5,3)
#print(x)
#x=torch.zeros(5,3,dtype=torch.long)
#print(x)
#x=torch.tensor([1,3.2])
#print(x)
#x=x.new_ones(3,2,dtype=torch.float64)
#print(x)
#x=torch.randn_like(x,dtype=torch.float)
#print(x)
#print(x.size())
#print(x.shape)
#print(torch.eye(3))
#print(torch.arange(1,5,2))
#print(torch.linspace(1,5,6))
## 均匀分布
#print(torch.rand(3,4))
## 标准分布
#print(torch.randn(3,4))
## 正态分布
#print(torch.normal(0.1,torch.arange(1,0,-0.1)))
## 均匀分布报错??????????????????????????????????????????
##print(torch.uniform(2,3))
#x=torch.rand(3,5)
##print(torch.non_zero(x))
#y=x.view(15)
#print(y)
#print(x.clone().view(-1,5))

##########day2###########

import torch
x=torch.ones(3,2,requires_grad=True)
y=x+2
print(y,y.grad_fn)
#LinearRegression
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

#构造数据集
num_inputs=2
num_examples=1000
true_w=[2,-3.4]
true_b=4.2
features=torch.from_numpy(np.random.normal(0,1,(num_examples,num_inputs)))
labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
labels+=torch.from_numpy(np.random.normal(0,0.01,size=labels.size()))
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    
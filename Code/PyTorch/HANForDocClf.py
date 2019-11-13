# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:13:55 2019

@author: ManSsssuper
"""

import torch

#squeeze，压缩，只有当维度为1才会去掉，不指定维度会去掉所有1维的维度
x=torch.zeros(2,1,2,1,2)
print(x.size())
y=torch.squeeze(x)
print(y.size())

#unsqueeze，解压缩，增加维度为1
y=torch.unsqueeze(y,0)
print(y.size())

##矩阵乘法torch.mm
x=torch.randn(3,2)
y=torch.randn(2,5)
z=torch.mm(x,y)
print(z)
#tensor.expand是扩展,只能当某一维度是1时可用，该方法不会改变内存，只会改变view
#某一维度是-1时，代表不要改变该维度
x=torch.randn(1,3)
y=x.expand(4,3)
print(x)
print(y)
y=x.expand(5,-1)
print(y)
z=x.expand_as(y)
print(z)

#torch.cat连接张量，在不同的维度上
x=torch.randn(2,3)
y=torch.cat((x,x,x),0)
print(y,y.shape)
y=torch.cat((x,x,x),1)
print(y,y.shape)

#torch.tensor.uniform_() 将tensor用从均匀分布中抽样得到的值填充。
print(torch.Tensor(2,3).uniform_(5,10))

#torch.Tensor.transpose()#将交换给定的尺寸dim0和dim1
a=torch.randn(2,3,4)
print(a)
b=a.transpose(0,2)
print(b)
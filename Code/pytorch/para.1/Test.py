# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:05:42 2019

@author: ManSsssuper
"""
import numpy as np
import torch
x=torch.empty(5,4)
print(x)
x=torch.rand(5,4)
print(x)
x=torch.zeros(5,3,dtype=torch.long)
print(x)
x=torch.tensor([1,2,3])
x=x.new_ones(5,3,dtype=torch.double)
print(x)
x=torch.randn_like(x,dtype=torch.float)
print(x)
print(x.size())
y=torch.rand(5,3)

#几种加法
print(x+y)
print(torch.add(x,y))
result=torch.empty(5,3)
torch.add(x,y,out=result)
print(result)
#任何 以``_`` 结尾的操作都会用结果替换原变量
y.add_(x)
print(y)

#索引
print(x[:,2])
#view相当于reshape
x=torch.randn(4,4)
y=x.view(16)
z=x.view(-1,8)
print(x,y,z)

#只有一个张量用item得到
x=torch.randn(1)
print(x)
print(x.item())

#numpy和tensor随意转换
#numpy和tensor的转换是基于内存地址的，因此，任意改变一方都会影响另一方
a=torch.ones(6,4)
b=a.numpy()
print(a,b)
a.add_(1)
print(a,b)

c=np.ones([6,4])
d=torch.from_numpy(c)
print(c,d)
np.add(c,1,out=c)
print(c,d)

#Tensor默认基于CPU，如果想把Tensor放到GPU上，可以执行如下操作
if torch.cuda.is_available():
    device=torch.device('cuda')
    #直接在GPU上创建张量
    y=torch.ones_like(x,device=device)
    #将Tensor移动到GPU上
    x=x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
    
#自动求导机制
import torch
x=torch.ones(2,2,requires_grad=True)
print(x)
y=x+2
print(y)
print(y.grad_fn)
z=y*y*3
out=z.mean()
print(z,out)
#.requires_grad_()改变张量的梯度属性
a=torch.randn(2,2)
a=((a*3)/(a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b=(a*a).sum()
print(b.grad_fn)

out.backward()

print(x.grad)
#autograd做更多的操作
x=torch.randn(3,requires_grad=True)
y=x*2
while y.data.norm()<1000:
    y=y*2
print(y)
gradients=torch.tensor([0.1,1.0,0.0001],dtype=torch.float)
y.backward(gradients)
print(x.grad)
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)
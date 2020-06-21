# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:54:51 2020

@author: ManSsssuper
    共享data的操作：
        x.view()
        from_numpy()
        numpy()
        y[:]=x+y
        torch.add(x,y,out=y)
    不共享内存
        x.clone()
        y=x+y
        y=torch.tensor(np.zeros(1))
"""
import torch
x=torch.empty(5,3)
x=torch.zeros(5,3,dtype=torch.long)
x=torch.tensor([5.5,3])
x=x.new_ones(5,3,dtype=torch.float64)
x=torch.randn_like(x,dtype=torch.float)
print(x.size(),x.shape)

x=torch.eye(5,2)#对角线为1
print(x)

#rand:均匀分布，randn标准正态分布
x=torch.rand(5,3)
print('rand',x)
x=torch.randn(5,3)
print('randn',x)

x,y=torch.rand(5,3),torch.rand(5,3)
print(x+y)
print(torch.add(x,y))
result=torch.empty(5,3)
torch.add(x,y,out=result)
print(result)
#inplace,后缀都是加_
y.add_(x)
print(y)

#索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改。
y=x[0,:]
y+=1
print(y)
print(x[0,:])

print(torch.index_select(x,0,torch.tensor([2,4])))
print(torch.masked_select(x,x>0))
print(torch.nonzero(x))

#view()，共享内存
y=x.view(15)
z=x.view(-1,5)
print(x,y,z)
x+=1
print(x,y)
#clone
x_cp=x.clone().view(15)
x-=1
print(x)
print(x_cp)
#使用clone还有一个好处是会被记录在计算图中，即梯度回传到副本时也会传到源Tensor

x=torch.randn(1)
print(x)
print(x.item())

#线性代数
x=torch.rand(5,3)
print(torch.trace(x))
print(torch.diag(x))
print(torch.triu(x))#上三角
print(torch.mm(x,torch.rand(3,1)))
print(torch.t(x))
print(torch.dot(x[0,:],torch.rand(3)))#一维向量
print(torch.inverse(x[:3,:]))
print(torch.svd_lowrank(x))

#广播
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)

#加法会开辟新内存
x=torch.rand(3,1)
y=torch.rand(3,1)
id_before=id(y)
y=x+y
print(id(y)==id_before)

#不开辟新内存的方法
x=torch.rand(3,1)
y=torch.rand(3,1)
id_before=id(y)
y[:]=x+y#torch.add(x,y,out=y)
print(id(y)==id_before)
#虽然view返回的Tensor与源Tensor是共享data的，但是依然是一个新的Tensor（因为Tensor除了包含data外还有一些其他属性），二者id（内存地址）并不一致。

#tensor和numpy转换
#共享内存
import numpy as np
y=np.zeros((3,1))
z=torch.from_numpy(y)
y+=1
print(y,z)
m=z.numpy()
z+=1
print(m,z)
#不共享内存
n=torch.tensor(y)
y+=1
print(y,n)
# 以下代码只有在PyTorch GPU版本上才会执行
if torch.cuda.is_available():
    device = torch.device("cuda")          # GPU
    y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
    x = x.to(device)                       # 等价于 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))# to()还可以同时更改数据类型    


#自动求梯度
"""
是requires_grad，不是require_grad
"""
x=torch.ones(2,2,requires_grad=True)
print(x,x.grad_fn)

y=x+2
print(y)
print(y.grad_fn)
#叶子节点
print(x.is_leaf,y.is_leaf)
z=y*y*3
out=z.mean()
print(z,out)
a = torch.randn(2, 2) # 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
print(a.requires_grad) # False
a.requires_grad_(True)
print(a.requires_grad) # True
b = (a * a).sum()
print(b.grad_fn)

#backward()
out.backward()
print(x.grad)

#反向传播梯度清0
out2=x.sum()
out2.backward()
print(x.grad)
out3=x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)

#梯度截断
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2 
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2

print(x.requires_grad)
print(y1, y1.requires_grad) # True
print(y2, y2.requires_grad) # False
print(y3, y3.requires_grad) # True

#要想修改tensor值，而不被追踪，修改x.data
x = torch.ones(1,requires_grad=True)

print(x.data) # 还是一个tensor
print(x.data.requires_grad) # 但是已经是独立于计算图之外

y = 2 * x
x.data *= 100 # 只改变了值，不会记录在计算图，所以不会影响梯度传播

y.backward()
print(x) # 更改data的值也会影响tensor的值
print(x.grad)
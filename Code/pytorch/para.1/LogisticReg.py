# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:57:57 2019

@author: ManSsssuper
"""
import torch.nn as nn
import numpy as np
import torch
data=np.loadtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric')
n,l=data.shape
for j in range(l-1):
    meanVal=np.mean(data[:,j])
    stdVal=np.std(data[:,j])
    data[:,j]=(data[:,j]-meanVal)/stdVal
np.random.shuffle(data)
train_data=data[:900,:l-1]
train_lab=data[:900,-1]-1
print(train_lab)
test_data=data[900:,:l-1]
test_lab=data[900:,-1]-1
class LR(nn.Module):
    def __init__(self):
        super(LR,self).__init__()
        self.fc=nn.Linear(24,2)
    def forward(self,x):
        out=self.fc(x)
        out=torch.sigmoid(out)
        return out
def test(pred,lab):
    t=pred.max(-1)[1]==lab
    return torch.mean(t.float())
net=LR()
criterion=nn.CrossEntropyLoss()
optm=torch.optim.Adam(net.parameters())
epochs=1000
for i in range(epochs):
    net.train()
    x=torch.from_numpy(train_data).float()
    y=torch.from_numpy(train_lab).long()
    yhat=net(x)
    loss=criterion(yhat,y)
    optm.zero_grad()
    loss.backward()
    optm.step()
    if (i+1)%100==0:
        net.eval()
        test_in=torch.from_numpy(test_data).float()
        test_l=torch.from_numpy(test_lab).long()
        test_out=net(test_in)
        print(test_out)
        # 使用我们的测试函数计算准确率
        accu=test(test_out,test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(i+1,loss.item(),accu))
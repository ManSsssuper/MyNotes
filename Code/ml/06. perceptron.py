# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 18:28:13 2020

@author: ManSsssuper
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

#读取数据到df
iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['label']=iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
print(df.label.value_counts())
#绘图
plt.scatter(df[:50]['sepal length'],df[:50]['sepal width'],label='0')
plt.scatter(df[50:100]['sepal length'],df[50:100]['sepal width'],label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
#处理数据为二分类，保留前两维特征，使用前100个样例
data=np.array(df.iloc[:100,[0,1,-1]])
X,y=data[:,:-1],data[:,-1]
y=np.array([1 if i==1 else -1 for i in y])

#perceptron模型部分
class perceptron：
    def __init__(self,input_dim):
        self.w=np.ones(input_dim,dtype=np.float32)
        self.b=0
        self.l_rate=0.1
    def sign(self,x,w,b):
        y=np.dot(x,w)+b
        return y
    def fit(self,X_train,y_train):
        is_wrong=False
        while not is_wrong:
            wrong_count=0
            for d in range(len(X_train)):
                X=X_train[d]
                y=y_train[d]
                if y*self.sign(X,self.w,self.b)<=0:
                    self.w=self.w+self.l_rate*np.dot(y,X)
                    self.b=self.b+self.l_rate*y
                    wrong_count+=1
            if wrong_count==0:
                is_wrong=True
        return 'train OK'

#训练
clf=perceptron(2)
clf.fit(X,y)

#绘图
x_points=np.linspace(4,7,10)
y_= -(perceptron.w[0]*x_points + perceptron.b)/perceptron.w[1]
plt.plot(x_points,y_)
plt.plot(data[:50,0],data[:50],1],'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()


#sklearn直接调用
from sklearn.linear_model import Perceptron
clf=Perceptron(fit_intercept=False,n_iter=1000,shuffle=False)
clf.fit(X,y)
#w和b
print(clf.coef_)
print(clf.intercept_)
x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()


























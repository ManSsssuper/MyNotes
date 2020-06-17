# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:17:07 2020

@author: ManSsssuper
    np.ravel():多维数组扁平为一维数组，但是返回的是视图，数据内存不变
    np.flatten():功能上同，但是返回的是新的对象
    pairwise_distances_argmin (X,Y):为X寻找距离最小的Y的索引
    np.where(array==value):返回array中数值==value的元素的index
"""
#使用EM算法进行iris的无监督分类
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import load_iris

iris=load_iris()
X=iris.data
y=iris.target
n_components = 3
feature_pairs=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]

for k,pair in enumerate(feature_pairs):
    x=X[:,pair]
    m = np.array([np.mean(x[y == i], axis=0) for i in range(3)])  # 均值的实际值
    print(m)
    gmm=GaussianMixture(n_components=n_components,covariance_type='full',random_state=2020)
    gmm.fit(x)
    print ('预测均值 = \n', gmm.means_)
    print ('预测方差 = \n', gmm.covariances_)
    y_hat = gmm.predict(x)
#    print(y_hat)
    y_hat = gmm.predict(x)
    order = pairwise_distances_argmin(m, gmm.means_, axis=1, metric='euclidean')
    print ('顺序：\t', order)
    """
        使用pairwise_distances_argmin计算实际均值与预测均值的最小距离
        实际顺序为：[0,1,2]，通过上述方法匹配到预测均值最近为[1,2,0]
        也就是说GMM的第1类对应着我们实际的第0类
            GMM的第2类对应着我们实际的第1类
            GMM的第0类对应着我们实际的第2类
        需要调整顺序才能计算准确率
    """
    y_change=np.zeros(y_hat.shape)
    for i in range(3):
#        print(np.where(y_hat==order[i]))
        y_change[np.where(y_hat==order[i])]=i
#    print(y_change)
    acc = u'准确率：%.2f%%' % (100*np.mean(y_change == y))
    print(acc)

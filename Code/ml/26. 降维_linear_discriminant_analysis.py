# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:45:31 2020

@author: ManSsssuper
"""
"""
        solver : 即求LDA超平面特征矩阵使用的方法。可以选择的方法有奇异值分解"svd"，
            最小二乘"lsqr"和特征分解"eigen"。一般来说特征数非常多的时候推荐使用svd，
            而特征数不多的时候推荐使用eigen。主要注意的是，如果使用svd，则不能指定正则化参数shrinkage
            进行正则化。默认值是svd
　　　　shrinkage：正则化参数，可以增强LDA分类的泛化能力。如果仅仅只是为了降维，
            则一般可以忽略这个参数。默认是None，即不进行正则化。可以选择"auto",
            让算法自己决定是否正则化。当然我们也可以选择不同的[0,1]之间的值进行交叉验证调参。
            注意shrinkage只在solver为最小二乘"lsqr"和特征分解"eigen"时有效。
　　　　priors ：类别权重，可以在做分类模型时指定不同类别的权重，进而影响分类模型建立。
            降维时一般不需要关注这个参数。
　　　　n_components：
            即我们进行LDA降维时降到的维数。在降维时需要输入这个参数。注意只能为[1,类别数-1)
            范围之间的整数。如果我们不是用于降维，则这个值可以用默认的None。

        只是为了降维，则只需要输入n_components,注意这个值必须小于“类别数-1”。PCA没有这个限制。
        一般来说，如果我们的数据是有类别标签的，那么优先选择LDA去尝试降维；
        当然也可以使用PCA做很小幅度的降维去消去噪声，然后再使用LDA降维。
        如果没有类别标签，那么肯定PCA是最先考虑的一个选择了。
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_classification
X, y = make_classification(n_samples=1000, n_features=3, n_redundant=0, n_classes=3, n_informative=2,
                           n_clusters_per_class=1,class_sep =0.5, random_state =10)
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
ax.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o',c=y)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
X_new = pca.transform(X)
print(X_new)
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o',c=y)
plt.show()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X,y)
X_new = lda.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o',c=y)
plt.show()
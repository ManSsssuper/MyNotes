# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:27:11 2020

@author: ManSsssuper

        min_weight_fraction_leaf : float, optional (default=0.)
                The minimum weighted fraction of the sum total of weights (of all
                the input samples) required to be at a leaf node. Samples have
                equal weight when sample_weight is not provided.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
iris = datasets.load_iris()
# 特征
iris_feature = iris.data
# 分类标签
iris_label = iris.target
# 划分
X_train, X_test, Y_train, Y_test = train_test_split(
    iris_feature, iris_label, test_size=0.3, random_state=42)
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
print("训练集:", rf.score(X_train, Y_train))
print("测试集:", rf.score(X_test, Y_test))

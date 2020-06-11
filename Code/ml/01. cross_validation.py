# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:07:12 2020

@author: ManSsssuper
"""
"""
    数据切分/交叉验证
    BaseCrossValidator
    KFold
    GroupKFold
    StratifiedKFold
    TimeSeriesSplit
    LeaveOneGroupOut
    LeaveOneOut
    LeavePGroupsOut
    LeavePOut
    RepeatedKFold
    RepeatedStratifiedKFold
    ShuffleSplit
    GroupShuffleSplit
    StratifiedShuffleSplit
    PredefinedSplit
    train_test_split
    check_cv
"""
"""
    train_test_split
        stratify:是否分层分割数据集，即是否保持训练集和测试集的类别比例保持一致
        分层采样后，效果飙升
        
    KFold:
        random_state:当固定随机种子时，shuffle仍然会打乱
        但是每次运行打乱的结果一致，所以在测试，做实验，模型效果对比应该固定随机种子
        当不shuffle时，此参数无效
        
    RepeatedKFold:
        没有参数shuffle，说明每一次k折都必然会打乱，否则没有意义
        
    ShuffleSplit:
        更像是放回随机采样，假设五折交叉，每一折都是随机采样然后放回，所以
        可能有两折是相同的，或者某一样本出现在多折当中
    分层采样：
        分层切割：train_test_split:stratify
        分层交叉验证：StratifiedKFold和StratifiedShuffleSplit
"""
# bootstrapping0.368

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedKFold
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm
data = pd.DataFrame(np.random.rand(10, 4), columns=list('ABCD'))
data['y'] = [random.choice([0, 1]) for i in range(10)]
print(data)
train = data.sample(frac=1.0, replace=True)
test = data.loc[data.index.difference(train.index)].copy()
print(train)
print(test)

# train_test_split
iris = datasets.load_iris()
print(iris.data.shape, iris.target.shape)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=2020, stratify=iris.target)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))


# kfold
X = np.random.rand(16, 2)
y = np.array(range(1, 17))
print(X.shape, y.shape)
kf = KFold(n_splits=5, shuffle=True, random_state=2020)
for train_index, test_index in kf.split(X):
    print('train_index', train_index, 'test_index', test_index)
    train_X, train_y = X[train_index], y[train_index]
    test_X = X[test_index], y[test_index]


# RepeatedKFold P次K折交叉验证
X = np.random.rand(16, 2)
y = np.array(range(1, 17))
print(X.shape, y.shape)
kf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=2020)
for train_index, test_index in kf.split(X):
    print('train_index', train_index, 'test_index', test_index)
    train_X, train_y = X[train_index], y[train_index]
    test_X = X[test_index], y[test_index]


# LeaveOneOut
loo = LeaveOneOut()
X = np.random.rand(4, 2)
y = np.array(range(1, 5))
for train_index, test_index in loo.split(X):
    print('train_index', train_index, 'test_index', test_index)


# LeavePOut
X = [1, 2, 3, 4]
lpo = LeavePOut(p=2)
for train_index, test_index in lpo.split(X):
    print('train_index', train_index, 'test_index', test_index)


# ShuffleSplit
X = np.random.rand(16, 2)
y = np.array(range(1, 17))
ss = ShuffleSplit(n_splits=4, random_state=2020, test_size=0.25)
for train_index, test_index in ss.split(X):
    print('train_index', train_index, 'test_index', test_index)


# 分层采样
# StratifiedKFold
X = np.array([[10, 1], [20, 2], [30, 3], [40, 4], [50, 5], [60, 6], [70, 7], [80, 8], [90, 9], [100, 10],
              [110, 1], [120, 2], [130, 3], [140, 4], [150, 5], [160, 6], [170, 7], [180, 8], [190, 9], [200, 10]])
# 五个类别：1:1:1:1:1
Y1 = np.array([1, 1, 2, 3, 3, 2, 4, 4, 5, 5, 1, 1, 2, 3, 3, 2, 4, 4, 5, 5])
# 两个类别：2:3
Y2 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2020)
for train_index, test_index in skf.split(X, Y2):
    print('train_index', train_index, 'test_index', test_index)


# StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5)
for train_index, test_index in skf.split(X, Y2):
    print('train_index', train_index, 'test_index', test_index)

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:07:12 2020

@author: ManSsssuper
"""
"""
    cross_val_score
    cross_val_predict
    cross_validate
    learning_curve
    permutation_test_score
    validation_curve
    
    cross_value_score:
        cv:可以指定cv，也可以直接指定int
        scoring：可以指定score
    cross_validate:
        方法和cross_validate_score有个两个不同点：它允许传入多个评估方法，可以使用两种方法
        来传入，一种是列表的方法，另外一种是字典的方法。
        最后返回的scores为一个字典，字典的key为：dict_keys(['fit_time', 'score_time', 
        'test_score', 'train_score'])
    cross_val_predict: 
        和 cross_val_score的使用方法是一样的，但是它返回的是一个使用交叉验证以后的输出值，
        而不是评分标准。它的运行过程是这样的，使用交叉验证的方法来计算出每次划分为测试集部分
        数据的值，知道所有的数据都有了预测值。假如数据划分为[1,2,3,4,5]份，它先用[1,2,3,4]
        训练模型，计算出来第5份的目标值，然后用[1,2,3,5]计算出第4份的目标值，直到都结束为止。
"""
# cross_val_score
from sklearn import datasets
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, recall_score
from sklearn import metrics
iris = datasets.load_iris()
clf = SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print(scores)

#自定义cv
my_cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
scores = cross_val_score(clf, iris.data, iris.target, cv=my_cv)
print(scores)
#用自己的scoring
scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro')
print(scores)


# cross_validate
iris = load_iris()
scoring = ['precision_macro', 'recall_macro']
clf = SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, iris.data, iris.target,
                        scoring=scoring, cv=5, return_train_score=False)
print(scores.keys())
print(scores['fit_time'])
print(scores['test_recall_macro'])

#字典形式，最终返回结果会自动加上test
scoring = {'prec_macro': 'precision_macro',
           'rec_micro': make_scorer(recall_score, average='macro')}
clf = SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, iris.data, iris.target,
                        scoring=scoring, cv=5, return_train_score=False)
print(scores.keys())
print(scores['test_rec_micro'])

#返回预测结果
# cross_val_pre
iris = load_iris()
clf = SVC(kernel='linear', C=1, random_state=0)
predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)
print(predicted)
print(metrics.accuracy_score(predicted, iris.target))

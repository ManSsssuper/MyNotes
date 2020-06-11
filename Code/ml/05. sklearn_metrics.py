# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:48:22 2020

@author: ManSsssuper

    'micro':Calculate metrics globally by counting the total true positives, false negatives and false positives.
    'micro':通过先计算总体的TP，FN和FP的数量，再计算F1
    'macro':Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    'macro':分布计算每个类别的F1，然后做平均（各类别F1的权重相同）
"""
#f1
from sklearn.metrics import f1_score,precision_recall_fscore_support
y_true = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4]
y_pred = [1, 1, 1, 0, 0, 2, 2, 3, 3, 3, 4, 3, 4, 3]
print(f1_score(y_true,y_pred,average='micro'))
print(f1_score(y_true,y_pred,average='macro'))
print(precision_recall_fscore_support(y_true,y_pred,labels=[0,1,2,3,4]))

#recall-precision
from sklearn.metrics import recall_score
print(recall_score(y_true,y_pred,labels=[1,2,3,4],average='micro'))
from sklearn.metrics import precision_score
print(precision_score(y_true,y_pred,labels=[1,2,3,4],average='micro'))


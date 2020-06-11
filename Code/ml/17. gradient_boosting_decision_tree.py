# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:19:28 2020

@author: ManSsssuper
"""
"""
    GBDT也是迭代，使用了前向分布算法，但是弱学习器限定了只能使用CART回归树模型
    n_estimators:
        迭代次数/弱学习器个数
    learning_rate：
        学习率：弱学习器权重缩减系数
    subsample：
        子采样，对特征采样，无放回采样，有别于bootstrapping
    init：
        f_0(x)
    loss:
        分类：对数似然损失（deviance）/指数损失（exponential）
        回归：ls/lad/huber/quantile
    alpha：
        回归损失为huber/quantile时使用
    调参过程：
        lr->n_estimators->max_depth+min_samples_split->min_samples_split+min_samples_leaf
        ->max_features->sub_samples->lr/2,n_e*2提升泛化能力
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split 
#数据导入
##先批量生成列名
names=[]
for i in range(34):
    names.append('Var'+str(i+1))
names.append('gbflag')
df = pd.read_csv(r'./data/ionosphere.data',header = None,names=names)
##查看前五行数据
print(df.head)
##查看数据有没有缺失值，异常值
df.describe()
x_columns = [x for x in df.columns if x not in 'gbflag']
X = df[x_columns]
y = df['gbflag']
#将数据集分成训练集，测试集
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)

#GBDT
##重要参数max_depth=4,max_features=10,n_estimators=80
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier()
gbdt.fit(X_train,y_train)
pred = gbdt.predict(X_test)
pd.crosstab(y_test,pred)

#算法评估指标
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(classification_report(y, pred, digits=4))

#交叉验证（数据样本少，可以使用交叉验证方法）
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
kf = KFold(n_splits = 10)
scores = []
for train,test in kf.split(X):
    train_X,test_X,train_y,test_y = X.iloc[train],X.iloc[test],y.iloc[train],y.iloc[test]
    gbdt =  GradientBoostingClassifier(max_depth=4,max_features=9,n_estimators=100)
    gbdt.fit(train_X,train_y)
    prediced = gbdt.predict(test_X)
    print(accuracy_score(test_y,prediced))
    scores.append(accuracy_score(test_y,prediced))   
##交叉验证后的平均得分
np.mean(scores)

#自动调参
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
gbdt = GradientBoostingClassifier()
cross_validation = StratifiedKFold(y,n_folds = 10)
parameter_grid = {'max_depth':[2,3,4,5],
                  'max_features':[1,3,5,7,9],
                  'n_estimators':[10,30,50,70,90,100]}
grid_search = GridSearchCV(gbdt,param_grid = parameter_grid,cv =cross_validation,
                           scoring = 'accuracy')
grid_search.fit(X,y)
#输出最高得分
grid_search.best_score_
#输出最佳参数
grid_search.best_params_

#########################################GBDT回归###############################
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=5, subsample=1
                                 , min_samples_split=2, min_samples_leaf=1, max_depth=3
                                 , init=None, random_state=None, max_features=None
                                 , alpha=0.9, verbose=0, max_leaf_nodes=None
                                 , warm_start=False
                                 )
train_feat = np.array([[1, 5, 20],
                       [2, 7, 30],
                       [3, 21, 70],
                       [4, 30, 60],
                       ])
train_id = np.array([[1.1], [1.3], [1.7], [1.8]]).ravel()
test_feat = np.array([[5, 25, 65]])
test_id = np.array([[1.6]])
print(train_feat.shape, train_id.shape, test_feat.shape, test_id.shape)
gbdt.fit(train_feat, train_id)
pred = gbdt.predict(test_feat)
total_err = 0
for i in range(pred.shape[0]):
    print(pred[i], test_id[i])
    err = (pred[i] - test_id[i]) / test_id[i]
    total_err += err * err
print(total_err / pred.shape[0])

#GBDT多分类
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

'''
调参：
loss：损失函数。有deviance和exponential两种。deviance是采用对数似然，exponential是指数损失，后者相当于AdaBoost。
n_estimators:最大弱学习器个数，默认是100，调参时要注意过拟合或欠拟合，一般和learning_rate一起考虑。
learning_rate:步长，即每个弱学习器的权重缩减系数，默认为0.1，取值范围0-1，当取值为1时，相当于权重不缩减。较小的learning_rate相当于更多的迭代次数。
subsample:子采样，默认为1，取值范围(0,1]，当取值为1时，相当于没有采样。小于1时，即进行采样，按比例采样得到的样本去构建弱学习器。这样做可以防止过拟合，但是值不能太低，会造成高方差。
init：初始化弱学习器。不使用的话就是第一轮迭代构建的弱学习器.如果没有先验的话就可以不用管
由于GBDT使用CART回归决策树。以下参数用于调优弱学习器，主要都是为了防止过拟合
max_feature：树分裂时考虑的最大特征数，默认为None，也就是考虑所有特征。可以取值有：log2,auto,sqrt
max_depth：CART最大深度，默认为None
min_sample_split：划分节点时需要保留的样本数。当某节点的样本数小于某个值时，就当做叶子节点，不允许再分裂。默认是2
min_sample_leaf：叶子节点最少样本数。如果某个叶子节点数量少于某个值，会同它的兄弟节点一起被剪枝。默认是1
min_weight_fraction_leaf：叶子节点最小的样本权重和。如果小于某个值，会同它的兄弟节点一起被剪枝。一般用于权重变化的样本。默认是0
min_leaf_nodes：最大叶子节点数
'''

gbdt = GradientBoostingClassifier(loss='deviance', learning_rate=1, n_estimators=5, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=2
                                  , init=None, random_state=None, max_features=None
                                  , verbose=0, max_leaf_nodes=None, warm_start=False
                                  )

train_feat = np.array([[6],
                       [12],
                       [14],
                       [18],
                       [20],
                       [65],
                       [31],
                       [40],
                       [1],
                       [2],
                       [100],
                       [101],
                       [65],
                       [54],
                       ])
train_label = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [2], [2], [2], [2]]).ravel()

test_feat = np.array([[25]])
test_label = np.array([[0]])
print(train_feat.shape, train_label.shape, test_feat.shape, test_label.shape)

gbdt.fit(train_feat, train_label)
pred = gbdt.predict(test_feat)
print(pred, test_label)


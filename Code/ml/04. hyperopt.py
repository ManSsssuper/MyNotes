# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:41:21 2020

@author: ManSsssuper
    fmin:最小化函数
        fn：函数
        space：调整的参数
        algo：参数搜索算法
            随机搜索
            Tree of Parzen Estimators（TPE）
            Adaptive TPE
        max_evals:搜索次数
    trials/STATUS_OK用来记录每一次搜索的结果，可以利用trials进行可视化
    xgboost_factory：模型工厂
    
"""
#####################hyperopt基础##########################
from hyperopt import fmin,tpe,hp
best=fmin(
        fn=lambda x:-(x-1)**2,
        space=hp.uniform('x',-2,2),
        algo=tpe.suggest,
        max_evals=100)
print(best)
###进阶
import hyperopt.pyll.stochastic
space = {
    'x': hp.uniform('x', 0, 1),
    'y': hp.normal('y', 0, 1),
    'name': hp.choice('name', ['alice', 'bob']),
}
print(hyperopt.pyll.stochastic.sample(space))


#使用trials
from hyperopt import fmin,tpe,hp,STATUS_OK,Trials

fspace={'x':hp.uniform('x',-5,5)}
def f(p):
    x=p['x']
    val=x**2
    return {'loss':val,'status':STATUS_OK}
trials=Trials()
best=fmin(fn=f,space=fspace,algo=tpe.suggest,max_evals=50,trials=trials)
print('best:', best)

print('trials:')
for trial in trials.trials[:2]:
    print(trial)
    
    
######################hyperopt-KNN###########################
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
iris=datasets.load_iris()
X=iris.data
y=iris.target
def KNN_factory(params):
    clf=KNeighborsClassifier(**params)
    return cross_val_score(clf,X,y).mean()
space={
       'n_neighbors': hp.choice('n_neighbors', range(1,100))
       }
def f(params):
    acc=KNN_factory(params)
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=100, trials=trials)
print('best:')
print(best)
################################hyperopt-SVM###################
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
iris=datasets.load_iris()
X=iris.data
y=iris.target
def SVM_factory(params):
    clf = SVC(**params)
    return cross_val_score(clf, X, y).mean()

space = {
    'C': hp.uniform('C', 0, 20),
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
    'gamma': hp.uniform('gamma', 0, 20),
}

def f(params):
    acc = SVM_factory(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=100, trials=trials)
print('best:')
print(best)
##########################Hyperopt xgboost###############################
from hyperopt import fmin, tpe, hp, partial
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn import datasets
iris=datasets.load_iris()
X=iris.data
y=iris.target
# 定义一个目标函数,接受一个变量,计算后返回一个函数的损失值，

#这里将xgb工厂和f合并了
def GBM_factory(params):
    max_depth = params["max_depth"] + 5
    n_estimators = params['n_estimators'] * 5 + 50
    learning_rate = params["learning_rate"] * 0.02 + 0.05
    subsample = params["subsample"] * 0.1 + 0.7
    min_child_weight = params["min_child_weight"]+1
    global attr_train, label_train

    gbm = xgb.XGBClassifier(nthread=4,  # 进程数
                            max_depth=max_depth,  # 最大深度
                            n_estimators=n_estimators,  # 树的数量
                            learning_rate=learning_rate,  # 学习率
                            subsample=subsample,  # 采样数
                            min_child_weight=min_child_weight,  # 孩子数
                            max_delta_step=10,  # 10步不降则停止
                            objective="binary:logistic")

    metric = cross_val_score(gbm, X, y).mean()
    print(metric)
    return -metric


# 定义参数的搜索空间
space = {"max_depth": hp.randint("max_depth", 15),
         # [0,1,2,3,4,5] -> [50,]
         "n_estimators": hp.randint("n_estimators", 10),
         # [0,1,2,3,4,5] -> 0.05,0.06
         "learning_rate": hp.randint("learning_rate", 6),
         # [0,1,2,3] -> [0.7,0.8,0.9,1.0]
         "subsample": hp.randint("subsample", 4),
         "min_child_weight": hp.randint("min_child_weight", 5),
         }
# 定义随机搜索算法。搜索算法本身也有内置的参数决定如何去优化目标函数
algo = partial(tpe.suggest, n_startup_jobs=1)
best = fmin(GBM_factory, space, algo=algo, max_evals=4)  # 对定义的参数范围，调用搜索算法，对模型进行搜索
print(best)
print(GBM_factory(best))

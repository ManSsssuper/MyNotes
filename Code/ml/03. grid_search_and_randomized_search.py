# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 08:59:58 2020

@author: ManSsssuper
"""
"""
    GridSearchCV
    RandomizedSearchCV
    ParameterGrid
    ParameterSampler
    fit_grid_point
"""
#############################GridSearch#################################
# 一般Gridsearch只在训练集上做k-fold并不会使用测试集.而是将测试集留在最后,
# 当gridsearch选出最佳模型的时候,在使用测试集测试模型的泛化能力.
# GridSearch

# Loading the Digits dataset
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # 网格搜索和随机搜索
from sklearn.neighbors import KNeighborsClassifier  # 要估计的是knn里面的参数，包括k的取值和样本权重分布方式
from sklearn.datasets import load_iris  # 自带的样本数据集
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# 将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# 设置gridsearch的参数
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# 设置模型评估的方法.如果不清楚,可以参考上面的k-fold章节里面的超链接
scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    # 构造这个GridSearch的分类器,5-fold
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_weighted' % score)
    # 只在训练集上面做k-fold,然后返回最优的模型参数
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    # 输出最优的模型参数
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    # 在测试集上测试最优的模型的泛化能力.
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


###############################RandomizedSearch#################

iris = load_iris()

X = iris.data  # 150个样本，4个属性
y = iris.target  # 150个类标号

k_range = range(1, 31)  # 优化参数k的取值范围
# 代估参数权重的取值范围。uniform为统一取权值，distance表示距离倒数取权值
weight_options = ['uniform', 'distance']
# 下面是构建parameter grid，其结构是key为参数名称，value是待搜索的数值列表的一个字典结构
# 定义优化参数字典，字典中的key值必须是分类算法的函数的参数名
param_grid = {'n_neighbors': k_range, 'weights': weight_options}
print(param_grid)

# 定义分类算法。n_neighbors和weights的参数名称和param_grid字典中的key名对应
knn = KNeighborsClassifier(n_neighbors=5)


# ================================网格搜索=======================================
# 这里GridSearchCV的参数形式和cross_val_score的形式差不多，其中param_grid是parameter grid所对应的参数
# GridSearchCV中的n_jobs设置为-1时，可以实现并行计算（如果你的电脑支持的情况下）
# 针对每个参数对进行了10次交叉验证。scoring='accuracy'使用准确率为结果的度量指标。可以添加多个度量指标
grid = GridSearchCV(estimator=knn, param_grid=param_grid,
                    cv=10, scoring='accuracy')
grid.fit(X, y)

print('网格搜索-度量记录：', grid.cv_results_)  # 包含每次训练的相关信息
print('网格搜索-最佳度量值:', grid.best_score_)  # 获取最佳度量值
print('网格搜索-最佳参数：', grid.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
print('网格搜索-最佳模型：', grid.best_estimator_)  # 获取最佳度量时的分类器模型

# 使用获取的最佳参数生成模型，预测数据
knn = KNeighborsClassifier(n_neighbors=grid.best_params_[
                           'n_neighbors'], weights=grid.best_params_['weights'])  # 取出最佳参数进行建模
knn.fit(X, y)  # 训练模型
print(knn.predict([[3, 5, 4, 2]]))  # 预测新对象

# =====================================随机搜索===========================================
rscv = RandomizedSearchCV(knn, param_grid, cv=10,
                          scoring='accuracy', n_iter=10, random_state=5)  #
rscv.fit(X, y)
print('随机搜索-度量记录：', grid.cv_results_)  # 包含每次训练的相关信息
print('随机搜索-最佳度量值:', grid.best_score_)  # 获取最佳度量值
print('随机搜索-最佳参数：', grid.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
print('随机搜索-最佳模型：', grid.best_estimator_)  # 获取最佳度量时的分类器模型
# 使用获取的最佳参数生成模型，预测数据
knn = KNeighborsClassifier(n_neighbors=grid.best_params_[
                           'n_neighbors'], weights=grid.best_params_['weights'])  # 取出最佳参数进行建模
knn.fit(X, y)  # 训练模型
print(knn.predict([[3, 5, 4, 2]]))  # 预测新对象
# =====================================自定义度量===========================================
# 自定义度量函数
def scorerfun(estimator, X, y):
    y_pred = estimator.predict(X)
    return metrics.accuracy_score(y, y_pred)

rscv = RandomizedSearchCV(knn, param_grid, cv=10,
                          scoring=scorerfun, n_iter=10, random_state=5)  #
rscv.fit(X, y)
print('随机搜索-最佳度量值:', grid.best_score_)  # 获取最佳度量值

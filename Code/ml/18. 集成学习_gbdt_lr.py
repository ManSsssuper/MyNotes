# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:25:14 2020

@author: ManSsssuper
    1）为什么要使用集成的决策树模型，而不是单棵的决策树模型：
        一棵树的表达能力很弱，不足以表达多个有区分性的特征组合，多棵树的表达能力更强一些。
        可以更好的发现有效的特征和特征组合
    2）为什么建树采用GBDT而非RF：RF也是多棵树，但从效果上有实践证明不如GBDT。
        且GBDT前面的树，特征分裂主要体现对多数样本有区分度的特征；后面的树，
        主要体现的是经过前N颗树，残差仍然较大的少数样本。优先选用在整体上有区分度的特征，
        再选用针对少数样本有区分度的特征，思路更加合理，这应该也是用GBDT的原因
    sklearn能设置树得个数，但不能设置叶子个数
    lgb能够设置叶子个数和节点个数
    如果gbdt得到得特征过于稀疏，可以进行特征选择后再用LR
"""
#sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier


gbm1 = GradientBoostingClassifier(n_estimators=50, random_state=10, subsample=0.6, max_depth=7,
                                  min_samples_split=900)
gbm1.fit(X_train, Y_train)
train_new_feature = gbm1.apply(X_train)
train_new_feature = train_new_feature.reshape(-1, 50)

enc = OneHotEncoder()

enc.fit(train_new_feature)

# # 每一个属性的最大取值数目
# print('每一个特征的最大取值数目:', enc.n_values_)
# print('所有特征的取值数目总和:', enc.n_values_.sum())

train_new_feature2 = np.array(enc.transform(train_new_feature).toarray())

#lightgbm
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 64,
    'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}


# number of leaves,will be used in feature transformation
num_leaf = 64

print('Start training...')
# train
gbm = lgb.train(params=params,
                train_set=lgb_train,
                valid_sets=lgb_train, )


print('Start predicting...')
# y_pred分别落在100棵树上的哪个节点上
y_pred = gbm.predict(x_train, pred_leaf=True)
y_pred_prob = gbm.predict(x_train)


result = []
threshold = 0.5
for pred in y_pred_prob:
    result.append(1 if pred > threshold else 0)
print('result:', result)


print('Writing transformed training data')
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[1]) * num_leaf],
                                       dtype=np.int64)  # N * num_tress * num_leafs
for i in range(0, len(y_pred)):
    # temp表示在每棵树上预测的值所在节点的序号（0,64,128,...,6436 为100棵树的序号，中间的值为对应树的节点序号）
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    # 构造one-hot 训练数据集
    transformed_training_matrix[i][temp] += 1

y_pred = gbm.predict(x_test, pred_leaf=True)
print('Writing transformed testing data')
transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[1]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    # 构造one-hot 测试数据集
    transformed_testing_matrix[i][temp] += 1

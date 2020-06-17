# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:58:40 2020

@author: ManSsssuper
"""
"""
    包里包括三种：svm.SVC(就是书上得那种O(N2))、svm.nuSVC、svm.linearSVC
    在 NuSVC/OneClassSVM/NuSVR 内的参数 nu ， 近似是训练误差和支持向量的比值。
    LinearSVC具有更大的灵活性在选择处罚和损失函数时，而且可以适应更大的数据集，
    他支持密集和稀疏的输入是通过一对一的方式解决的
    
    样本数量远小于特征数量：这种情况，利用情况利用linear核效果会高于RBF核。
    样本数量和特征数量一样大：线性核合适，且速度也更快。liblinear更适合
    样本数量远大于特征数量： 非线性核RBF等合适。
    如果特征数量比样本数量大得多,在选择核函数时要避免过拟合,而且正则化项是非常重要的.
    
    C：C-SVC的惩罚参数C?默认值是1.0，C越大，相当于惩罚松弛变量，希望松弛变量接近0，
        即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但
        泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
    kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
      　　0 – 线性：u\'v
     　　 1 – 多项式：(gamma*u\'*v + coef0)^degree
      　　2 – RBF函数：exp(-gamma|u-v|^2)
      　　3 – sigmoid：tanh(gamma*u\'*v + coef0)
    degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
    gamma ： ‘rbf’,‘poly’和‘sigmoid’的核函数参数。默认是’auto’，如果是auto，则值为1/n_features
        gamma越大，支持向量越少，。gamma值越小，支持向量越多
    coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
    probability ：是否采用概率估计？.默认为False
    shrinking ：是否采用shrinking heuristic方法，默认为true
    tol ：停止训练的误差值大小，默认为1e-3
    cache_size ：核函数cache缓存大小，默认为200
    class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
    verbose ：允许冗余输出
    max_iter ：最大迭代次数。-1为无限制。
    decision_function_shape ：‘ovo’, ‘ovr’ or None, default=ovr
    random_state ：数据洗牌时的种子值，int值
"""
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn import svm
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
svm_classifier = svm.SVC(
    C=1.0, kernel='rbf', decision_function_shape='ovr', gamma=0.01)
svm_classifier.fit(X_train, Y_train)
print("训练集:", svm_classifier.score(X_train, Y_train))
print("测试集:", svm_classifier.score(X_test, Y_test))

"""
    当数据量很大时，使用内核近似步骤
    首先使用内核近似将输入空间映射到特征空间
    再使用LinearSVC或者SGDClassifier进行分类
    SGDClassifier
        它用的是mini-batch来做梯度下降，在处理大数据的情况下收敛更快。
        对于特别大的数据还是优先使用SGDClassifier，其他的线性可能很慢或者直接跑不动。
        Linear classifiers (SVM, logistic regression, a.o.) with SGD training.
        通过选择loss来选择不同模型，hinge是SVM，log是LR
"""
X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 0, 1, 1]
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)
clf = SGDClassifier()
clf.fit(X_features, y)
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
              eta0=0.0, fit_intercept=True, l1_ratio=0.15,
              learning_rate='optimal', loss='hinge', max_iter=5, n_iter=None,
              n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
              shuffle=True, tol=None, verbose=0, warm_start=False)
clf.score(X_features, y)

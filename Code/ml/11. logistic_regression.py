import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
iris=datasets.load_iris()
X=np.array(iris.data)[:100]
y=np.array(iris.target)[:100]
print(X.shape)
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=2020)
#train_y=np.reshape(train_y,(train_y.shape[0],1))
#test_y=np.reshape(test_y,(test_y.shape[0],1))
class logistic_regression:
    def __init__(self,input_dim,epochs,lr):
        self.w=np.ones((input_dim,1))
        self.epochs=epochs
        self.lr=lr
    def sigmoid(self,y):
        s=1.0/(1.0+np.exp(-y))
        return s
    def loss(self,y_hat,y):
        loss=(-1.0 / X.shape[0]) * np.sum(y.T * np.log(y_hat) + (1 - y).T * np.log(1 - y_hat))  # 损失函数
        return loss
    def fit(self,X,y):
        epoch=0
        while epoch<self.epochs:
            y_hat=np.reshape(self.sigmoid(np.dot(X,self.w)),(X.shape[0]))
#            print(y_hat.shape)
            
            loss=self.loss(y_hat,y)
            grad = (1.0/X.shape[0])*np.dot(X.T,np.reshape(y_hat-y,(y.shape[0],1))) #损失函数的梯度
            
            l_w = self.w #上一轮迭代的参数
            self.w = self.w - self.lr*grad #参数更新
            y_hat_new=self.sigmoid(np.dot(X,self.w))
            loss_new = self.loss(y_hat_new,y)#当前损失值
            if abs(loss_new-loss)<1e-4:#终止条件
                break
            epoch += 1
        print('迭代到第{}次，结束迭代！'.format(epoch))
    def score(self,X,y):
        result = []
        for i in range(X.shape[0]):
            proba = self.sigmoid(np.dot(X[i,:], self.w))
            if proba < 0.5:
                preict =  0
            else:
                preict = 1
            result.append(preict)
        acc = (np.array(result)==y).mean()
        return acc
clf=logistic_regression(4,1000,0.1)
clf.fit(train_X,train_y)
print(clf.score(test_X,test_y))

"""
    LogisticRegression， LogisticRegressionCV 和logistic_regression_path
        其中LR和LR-CV的主要区别是LR-CV使用了交叉验证来选择正则化系数C
        而LR需要自己每次指定一个正则化系数
        LR-path为拟合数据选择合适逻辑回归的系数和正则化系数。主要是用在模型选择的时候
        RandomizedLogisticRegression主要是用L1正则化的逻辑回归来做特征选择
        penalty :
            str, 'l1' or 'l2', default: 'l2'
        solver : 
            {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},default: 'liblinear'
            除了liblinear可以用于L1，其余得只能用L2
            saga在样本量很大时使用
        multi_class : 
            str, {'ovr', 'multinomial'}, default: 'ovr'
            multinomial时，liblinear不能使用
        class_weight : 
            dict or 'balanced', default: None
            类别不平衡时使用，为不同得类设置不同得权重，高权重得类会获得较高得代价
        sample_weight：
            样本权重参数
            class_weight*sample_weight.
        dual:
            用来指明是否将原问题改成他的对偶问题，样本少时可以用
        tol:
            残差收敛条件，默认是0.0001，也就是只需要收敛的时候两步只差＜0.0001
        C：
            正则化系数得倒数，默认1，越小，正则化越强
        fit_intercept：
            使用b，默认true
        warm_start:
            是否使用上次的模型结果作为初始化，默认是False，表示不使用
        n_jobs:
            并行运算数量(核的数量)，默认为1，如果设置为-1，则表示将电脑的cpu全部用上。
        系数
            coef_: 变量中的系数。shape (1, n_features) or (n_classes, n_features)
            intercept_：截距。shape (1,) or (n_classes,)
            n_iter_ ：所有类的实际迭代次数。shape (n_classes,) or (1, )
        method
            decision_function(X)：预测样本的 confidence scores
            densify()：将系数矩阵转化成密集矩阵的格式
            sparsify()：将系数矩阵转换成稀疏矩阵格式
"""
####L=sklearn实现#####################
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(train_X,train_y)
print(clf.score(test_X,test_y))
print(clf.get_params())





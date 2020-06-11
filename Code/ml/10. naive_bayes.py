"""
    GaussianNB:
        适用于特征多为连续值得情况，假设特征服从高斯分布
        priors：类别先验，若为None，则p(y=ck)=mck/m
    MultinomialNB:
        多项式分布适用于多为离散值得时候，对应于李航书上得NB
        alpha：对应于书中得λ，λ=1时，即为Laplace平滑，可以>1
        fit_prior:
            是否对类别使用先验，若false，则p(y=ck)=1/k
            若true，若class_prior是None，则p(y=ck)=mck/m
            若true，若class_prior不是None，则按照给定得class_prior
    BernoulliNB：
        三个参数上同
        binarize：为None，默认每个特征为二元特征
            否则以其值将特征划分为两个部分
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
iris=datasets.load_iris()
X=iris.data()
y=iris.target()
clf=GaussianNB()
train_X,train_y,test_X,test_y=train_test_split(X,y,test_size=0.2,random_state=2020)
clf.fit(train_X,train_y)
clf.score(test_X,test_y)

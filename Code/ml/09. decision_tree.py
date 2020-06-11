#！／user/bin/env python
#-*- coding:utf-8 -*-

"""
    SKLEARN决策树具体实现方法
        其实sklearn的决策树仅仅是实现了cart树而已，当信息计算方式为Gini，Entropy，就用来分类
        注意，sklearn并没有实现ID3，但是cart中使用entropy的效果，等效于“二叉树的ID3”
        因为ID3可以是“二叉决策树”，也可以是“多叉决策树”，所以sklearn使用决策树+entropy方式时，
        无法实现“基于ID3算法的多叉决策树”
    sklearn树的剪枝
        6. max_depth：
            限制树的最大深度决策树多生长一层，对样本量的需求会增加一倍，所以限制树深度能够有效地
            限制过拟合。在高维度低样本量时非常有效；建议从=3开始尝试。
        7. min_samples_leaf：
            一个节点在分枝后，每个子节点都必须至少包含的训练样本数量一个节点在分枝后，
            每个子节点都必须包含至少min_samples_leaf个训练样本，两种取值：
            （1）整数
            （2）浮点型：如果叶节点中含有的样本量变化很大，输入浮点数表示样本量的百分比。
            如果分支后的子节点不满足参数条件，分枝就不会发生，或者，分枝会朝着满足每个子节点都
            包含min_samples_leaf个样本的方向去发生。这个参数可以保证每个叶子的最小尺寸，
            在回归问题中避免低方差，过拟合的叶子节点出现。
            搭配max_depth使用，在回归树中可以让模型变得更加平滑；建议从=5开始；
            对于类别不多的分类问题，=1通常就是最佳选择。
        8. min_samples_split：
            一个节点必须要至少包含的训练样本数量如果大于于这个数量，这个节点才允许被分枝，否则分枝就不会发生。
    树的精修
        9. max_features：
            分枝时考虑的最大特征个数。即在分支时，超过限制个数的特征都会被舍弃。
            但是在不知道决策树中的各个特征的重要性的情况下，强行设定这个参数可能会导致模型学习不足。
        10. min_impurity_decrease：
            子父节点信息增益的最小值。信息增益是父节点的信息熵与子节点信息熵之差，
            信息增益越大，说明这个分支对模型的贡献越大；相反的，如果信息增益非常小，
            则说明该分支对模型的建立贡献不大。又由于分支需要的计算量又非常大，
            所以如果信息增益非常小时，我们就选择放弃该分支。min_impurity_decrease 参数就是一个节点
            在分支时，与其子节点的信息增益最小值。

"""
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
#3：1拆分数据集
from sklearn.model_selection import train_test_split
#乳腺癌数据集
from sklearn.datasets import load_breast_cancer
import pydot
cancer = load_breast_cancer()
#参数random_state是指随机生成器，0表示函数输出是固定不变的
X_train,X_test,y_train,y_test = train_test_split(cancer['data'],cancer['target'],random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)
print('Train score:{:.3f}'.format(tree.score(X_train,y_train)))
print('Test score:{:.3f}'.format(tree.score(X_test,y_test)))
#生成可视化图
export_graphviz(tree,out_file="tree.dot",class_names=['严重','轻微'],feature_names=cancer.feature_names,impurity=False,filled=True)
#展示可视化图
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')



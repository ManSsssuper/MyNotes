# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:28:02 2020

@author: ManSsssuper
"""
import tensorflow as tf
import time
from sklearn.metrics import roc_auc_score

def read_data(data_file):
    x=[]
    y=[]
    lines=open(data_file)
    for line in lines:
        splits=line.split(" ")
        y.append(float(splits[0]))
        x.append([float(splits[1][2:]),float(splits[2][2:])])
    return (x,y)

x=tf.placeholder(tf.float32,shape=[None,2])
y=tf.placeholder(tf.float32,shape=[None])

m=4
learning_rate=0.3
u=tf.Variable(tf.random_normal([2,m], 0.0, 0.5),name='u')
w=tf.Variable(tf.random_normal([2,m], 0.0, 0.5),name='u')

U=tf.matmul(x,u)
p1=tf.nn.softmax(U)

W=tf.matmul(x,w)
p2=tf.nn.sigmoid(W)

pred=tf.reduce_sum(tf.multiply(p1,p2),1)

cost1=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
#cost2_1=tf.reduce_sum(tf.abs(u))
#cost2_2=tf.reduce_sum(tf.abs(w))
#cost=tf.add_n([cost1,cost2_1,cost2_2])
cost=tf.add_n([cost1])
train_op = tf.train.FtrlOptimizer(learning_rate).minimize(cost)
#train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess = tf.Session()
sess.run(init_op)
data_x,data_y=read_data('train.txt')
import pdb
#pdb.set_trace()
time_s=time.time()
for epoch in range(0,10000):
    f_dict = {x:data_x, y:data_y}
    _, cost_, predict_= sess.run([train_op, cost, pred],feed_dict=f_dict)
    auc=roc_auc_score(data_y, predict_)
    time_t=time.time()
    if epoch % 100 == 0:
        print("%d %ld cost:%f,auc:%f" % (epoch, (time_t-time_s),cost_,auc))


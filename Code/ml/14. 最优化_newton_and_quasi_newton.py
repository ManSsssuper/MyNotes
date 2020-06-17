# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:27:19 2020

@author: ManSsssuper
"""
"""
Newton法
Rosenbrock函数
函数 f(x)
梯度 g(x)
hessen 矩阵
"""

import numpy as np
import matplotlib.pyplot as plt

# 一阶导
def jacobian(x):
    return np.array([2*x[0]+3,2*x[1]+4])

# 二阶导
def hessian(x):
    return np.array([[2,0],[0,2]])

X1=np.arange(-1.5,1.5+0.05,0.05)
X2=np.arange(-3.5,2+0.05,0.05)
[x1,x2]=np.meshgrid(X1,X2)
f=x1**2+x2**2+3*x1+4*x2-26; # 给定的函数
plt.contour(x1,x2,f,20) # 画出函数的20条轮廓线


def newton(x0):

    print('初始点为:')
    print(x0,'\n')
    W=np.zeros((2,10**3))
    i = 1
    imax = 1000
    W[:,0] = x0 
    x = x0
    delta = 1

    while i<imax and delta>0.1:
        p = -np.dot(np.linalg.inv(hessian(x)),jacobian(x))
        print(jacobian(x))
        print(hessian(x))
        x0 = x
        x = x + p
        W[:,i] = x
        delta = sum((x-x0))
        print('第'+str(i)+'次迭代结果:')
        print(x,'\n')
        i=i+1
    W=W[:,0:i]  # 记录迭代点
    return W

x0 = np.array([1,1])
W=newton(x0)

plt.plot(W[0,:],W[1,:],'g*',W[0,:],W[1,:]) # 画出迭代点收敛的轨迹
plt.show()
#DFP
import numpy as np

#函数表达式
fun = lambda x:100*(x[0]**2 - x[1]**2)**2 +(x[0] - 1)**2

#梯度向量
gfun = lambda x:np.array([400*x[0]*(x[0]**2 - x[1]) + 2*(x[0] - 1),-200*(x[0]**2 - x[1])])

#Hessian矩阵
hess = lambda x:np.array([[1200*x[0]**2 - 400*x[1] + 2,-400*x[0]],[-400*x[0],200]])

def dfp(fun,gfun,hess,x0):
    #功能：用DFP算法求解无约束问题：min fun(x)
    #输入：x0式初始点，fun,gfun，hess分别是目标函数和梯度,Hessian矩阵格式
    #输出：x,val分别是近似最优点，最优解，k是迭代次数
    maxk = 1e5
    rho = 0.05
    sigma = 0.4
    epsilon = 1e-5 #迭代停止条件
    k = 0
    n = np.shape(x0)[0]
    #将Hessian矩阵初始化为单位矩阵
    Hk = np.linalg.inv(hess(x0))

    while k < maxk:
        gk = gfun(x0)
        if np.linalg.norm(gk) < epsilon:
            break
        dk = -1.0*np.dot(Hk,gk)
#         print dk

        m = 0;
        mk = 0
        while m < 20:#用Armijo搜索步长
            if fun(x0 + rho**m*dk) < fun(x0) + sigma*rho**m*np.dot(gk,dk):
                mk = m
                break
            m += 1
        #print mk
        #DFP校正
        x = x0 + rho**mk*dk
        print("第"+str(k)+"次的迭代结果为："+str(x))
        sk = x - x0
        yk = gfun(x) - gk

        if np.dot(sk,yk) > 0:
            Hy = np.dot(Hk,yk)
            sy = np.dot(sk,yk) #向量的点积
            yHy = np.dot(np.dot(yk,Hk),yk) #yHy是标量
            Hk = Hk - 1.0*Hy.reshape((n,1))*Hy/yHy + 1.0*sk.reshape((n,1))*sk/sy

        k += 1
        x0 = x
    return x0,fun(x0),k

x0 ,fun0 ,k = dfp(fun,gfun,hess,np.array([0,0]))
print(x0,fun0,k)
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:42:03 2020

@author: ManSsssuper
        支持随机访问(O(1))，但是增加和删除操作效率就低一些(平均时间复杂度O(n))
        python内置list就是动态数组
"""
import ctypes
class DynamicArray:
    def __init__(self):
        self._n=0
        self._capacity=10
        self._A=self._make_array(self._capacity)
        
    def _make_array(self,c):
        return (c*ctypes.py_object)()
    def __len__(self):
        return self._n
    def is_empty(self):
        return self._n==0
    def __getitem__(self,k):#O(1)
        if not 0<=k<self._n:
            raise ValueError('invalid index')
        return self._A[k]
    def append(self,obj):#O(1)
        if self._n==self._capacity:
            self._resize(2*self._capacity)
        self._A[self._n]=obj
        self._n+=1
    def _resize(self,c):
        B=self._make_array(c)
        for k in range(self._n):
            B[k] = self._A[k]
        self._A = B
        self._capacity = c  
    def insert(self, k, value):#O(n)
        if self._n == self._capacity:
            self._resize(2 * self._capacity)
        for j in range(self._n, k, -1):    #从后往前一个一个往后移
            self._A[j] = self._A[j-1]
        self._A[k] = value
        self._n += 1
    # O(n)    
    def remove(self, value):
        for k in range(self._n):
            if self._A[k] == value:     #一个个查value
                for j in range(k, self._n - 1):
                    self._A[j] = self._A[j+1]   ##再一个个移上来
                self._A[self._n - 1] = None
                self._n -= 1
                return
        raise ValueError( 'value not found' )
    def _print(self):
        for i in range(self._n):
            print(self._A[i], end = ' ')
        print()
mylist = DynamicArray()
print ('size was: ', str(len(mylist)))
mylist.append(10)
mylist.append(20)
mylist.append(30)
mylist.insert(0, 0)
mylist.insert(1, 5)
mylist.insert(3, 15)
mylist._print()
mylist.remove(20)
mylist._print()
print ('size is: ', str(len(mylist)))
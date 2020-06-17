# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:57:56 2020

@author: hp
    栈后进先出
    顺序表实现扩展存储空间操作复杂，且需要大块的存储空间；采用链表实现在这两个问题上有优势，但是链表实现更多的依赖与解释器的存储管理，会带来操作开销。
"""
#栈的列表实现
class Stack(object):
    def __init__(self):
        self.__list=[]
    def is_empty(self):
        return self.__list==[]
    def push(self,item):
        self.__list.append(item)
    def pop(self):
        if self.is_empty():
            raise Exception('emtpy stack')
        else:
            return self.__list.pop()
    def top(self):
        if self.is_empty():
            return 
        else:
            return self.__list[-1]

#栈的链表实现
class Node(object):
    def __init__(self,val):
        self.val=val
        self.next=None
class Stack2(object):
    def __init__(self):
        self.__head=None
    def is_empty(self):
        return self.__head is None
    def push(self,item):#后来push进来的做head
        node=Node(item)
        node.next=self.__head
        self.__head=node
    def pop(self):
        if self.is_empty():
            raise Exception('emtpy stack')
        else:
            p=self.__head
            self.__head=p.next
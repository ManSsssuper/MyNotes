# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:26:25 2020

@author: ManSsssuper
"""
# 最大堆
"""
    索引为i的节点
        parent = floor((i-1) / 2)    # 取整
        leftchild = 2 * i + 1
        rightchild = 2 * i + 2
    self._elements[ndx], self._elements[parent] = self._elements[parent], self._elements[ndx]
"""
# 自定义数组array




import heapq
import numpy as np
class Array(object):
    def __init__(self, size=32):
        self._size = size
        self._items = [None]*size

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        self._items[index] = value

    def __len__(self):
        return self._size

    def clear(self, value=None):
        for i in range(self._size):
            self._items[i] = value

    def __iter__(self):
        for item in self._items:
            yield item

# 最大堆：父节点永远大于子节点


class MaxHeap(object):
    def __init__(self, maxsize=None):
        self.maxsize = maxsize
        self._elements = Array(maxsize)
        self._count = 0

    def __len__(self):
        return self._count

    def add(self, value):
        if self._count >= self.maxsize:
            raise Exception('full')
        self._elements[self._count] = value
        self._count += 1
        self._siftup(self._count-1)

    def remove(self):
        if self._count <= 0:
            raise Exception('empty')
        value = self._elements[0]
        self._count -= 1
        self._elements[0] = self._elements[self._count]
        self._siftdown(0)
        self._elements[self._count] = None
        return value

    def _siftup(self, ndx):
        if ndx > 0:
            # parent=int((ndx-1)/2)
            parent = np.floor((ndx-1)/2)
            if self._elements[ndx] > self._elements[parent]:
                self._elements[ndx], self._elements[parent] = self._elements[parent], self._elements[ndx]
                self._siftup(parent)

    def _siftdown(self, ndx):
        leftchild = 2 * ndx + 1
        rightchild = 2*ndx + 2
        larger = ndx
        if leftchild < self._count and self._elements[leftchild] > self._elements[larger]:
            larger = leftchild
        if rightchild < self._count and self._elements[rightchild] > self._elements[larger]:
            larger = rightchild
        if larger != ndx:
            self._elements[ndx], self._elements[larger] = self._elements[larger], self._elements[ndx]
            self._siftdown(larger)


#####################python内置实现###########################
'''
    heaqp模块提供了堆队列算法的实现，也称为优先级队列算法。
    要创建堆，请使用初始化为[]的列表，或者可以通过函数heapify（）将填充列表转换为堆。
    提供以下功能：
    heapq.heappush（堆，项目）
    将值项推入堆中，保持堆不变。
    heapq.heapify（x）
    在线性时间内将列表x转换为堆。
    heapq.heappop（堆）
    弹出并返回堆中的最小项，保持堆不变。如果堆是空的，则引发IndexError。
'''

# 1 heappush生成堆+ heappop把堆从小到大pop出来
heap = []
data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
for i in data:
    heapq.heappush(heap, i)
print(heap)

lis = []
while heap:
    lis.append(heapq.heappop(heap))
print(lis)

# 2 heapify生成堆+ heappop把堆从小到大pop出来
data2 = [1, 5, 3, 2, 9, 5]
heapq.heapify(data2)
print(data2)

lis2 = []
while data2:
    lis2.append(heapq.heappop(data2))
print(lis2)

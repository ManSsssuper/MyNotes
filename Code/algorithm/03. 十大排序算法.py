# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:26:20 2020

@author: ManSsssuper
        https://www.cxyxiaowu.com/725.html
        
"""

"""
        # 冒泡排序O(N^2)
        # 两层for循环，比较相邻值大小
        # 从小到大排序
"""


def sort_maopao(arr):
    for i in range(len(arr)-1):
        is_sorted = True  # 如果有序，跳出循环
        for j in range(len(arr)-1-i):
            temp = 0
            if arr[j] > arr[j+1]:
                temp = arr[j]
                arr[j] = arr[j+1]
                arr[j+1] = temp
                is_sorted = False
                # arr[j],arr[j+1]=arr[j+1],arr[j]并不会排序
        if is_sorted:
            break
    return arr


print(sort_maopao([2, 3, 1, 5]))

"""
        # 选择排序O(N^2)
        # 两层循环，每次循环选择最小值拎出来与当前位置交换
"""


def sort_xuanze(arr):
    for i in range(len(arr)-1):  # 最后一个元素不需要比较
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        if min_idx != i:
            temp = arr[min_idx]
            arr[min_idx] = arr[i]
            arr[i] = temp
    return arr


print(sort_xuanze([2, 3, 1, 5]))

"""
        # 插入排序最好情况O(N),最差O(N^2)
        # 每次从数组里抽出一个元素，假设该元素前面的都排好序了，后面的都乱，
        # 将该元素与前面的元素依次比较，找到合适的位置插入
        # 确实像玩扑克牌
"""


def sort_charu(arr):
    for i in range(len(arr)):  # 将i插入到0到i-1中
        temp = arr[i]
        insert = 0  # 标志位
        for j in range(i-1, -1, -1):
            if temp < arr[j]:
                arr[j+1] = arr[j]  # 向后移动数据
            else:
                insert = j+1
                break
        # 如果没有进入到else里面，insert=0说明temp是当前比较最小值
        arr[insert] = temp
    return arr


print(sort_charu([2, 3, 1, 5]))

"""
    # 希尔排序,在插入排序上进阶
    #https://blog.csdn.net/qq_39207948/article/details/80006224
"""


def sort_shell(arr):
    gap = int(len(arr)/2)
    while gap > 0:
        for i in range(gap, len(arr)):
            # 内层就是插入排序了
            temp = arr[i]
            insert = i % gap
            for j in range(i-gap, -1, -gap):
                if arr[j] > arr[i]:
                    arr[j+gap] = arr[j]
                else:
                    insert = j+gap
                    break
            arr[insert] = temp
#        print(arr)
        gap = int(gap/2)
    return arr


print(sort_shell([2, 3, 1, 5]))
"""
    # 归并排序O(Nlog(N)),空间复杂度O(1)，因为辅助数组在一开始就申请好了
    # 归并法属于分治的思想
"""


def sort_guibing(arr):
    def merge(arr, temp_arr, start_idx, middle_idx, end_idx):
        temp_arr[start_idx:end_idx+1] = arr[start_idx:end_idx+1]
        left = start_idx
        right = middle_idx+1
        for i in range(start_idx, end_idx+1):
            if left > middle_idx:  # 左半部分插入完成，只剩下右半部分
                arr[i] = temp_arr[right]
                right += 1
            elif right > end_idx:  # 右边半部分插入完成，只剩下左半部分
                arr[i] = temp_arr[left]
                left += 1
            elif temp_arr[left] < temp_arr[right]:  # 左指针是最小值
                arr[i] = temp_arr[left]
                left += 1
            else:  # 右指针是最小值
                arr[i] = temp_arr[right]
                right += 1

    def sort(arr, temp_arr, start_idx, end_idx):
        if end_idx <= start_idx:
            return
        middle_idx = int(start_idx+(end_idx-start_idx)/2)
        sort(arr, temp_arr, start_idx, middle_idx)
        sort(arr, temp_arr, middle_idx+1, end_idx)
        merge(arr, temp_arr, start_idx, middle_idx, end_idx)
#        print(arr)

    temp_arr = [None]*len(arr)
    sort(arr, temp_arr, 0, len(arr)-1)
    return arr


print(sort_guibing([2, 3, 1, 5, 5, 1, 23, 6, 8, 12, 434]))

"""
    # 快速排序
    快速排序的时间复杂度是 O(nlogn)，极端情况下会退化成 O(n2)，
    为了避免极端情况的发生，选取基准值应该做到随机选取，或者是打乱一下数组再选取。
    另外，快速排序的空间复杂度为 O(1)。
    https://www.runoob.com/w3cnote/quick-sort.html
    # 以第一个数为基准数
"""


def sort_kuaisu(arr):
    def quick_sort(arr, l, r):  # 递归调用
        if l < r:
            left, right, temp = l, r, arr[l]
            while left < right:
                while left < right and arr[right] > temp:  # 从右往左
                    right -= 1
                if left < right:
                    arr[left] = arr[right]
                    left += 1
                while left < right and arr[left] < temp:  # 从左往右
                    left += 1
                if left < right:
                    arr[right] = arr[left]
                    right -= 1
            arr[left] = temp  # 基准数据填入
            quick_sort(arr, l, left-1)
            quick_sort(arr, left+1, r)
    # 主程序
    quick_sort(arr, 0, len(arr)-1)
    return arr


print(sort_kuaisu([2, 3, 1, 5, 5, 1, 23, 6, 8, 12, 434]))

"""
    # 堆排序O(Nlog(N))
    # 自定义最大堆
    # 最大堆：父节点永远大于子节点
"""


class MaxHeap(object):
    def __init__(self, maxsize=None):
        self.maxsize = maxsize
        self._elements = [None]*self.maxsize
        self._count = 0

    def __len__(self):
        return self._count

    def add(self, value):
        if self._count >= self.maxsize:
            raise Exception('full')
        self._elements[self._count] = value
        self._count += 1
        self._siftup(self._count-1)
#        print(self._elements)

    def remove(self):
        if self._count <= 0:
            raise Exception('empty')
        value = self._elements[0]
        self._count -= 1
        self._elements[0] = self._elements[self._count]
        self._siftdown(0)
        self._elements[self._count] = None
#        print(self._elements)
        return value

    def _siftup(self, ndx):
        if ndx > 0:
            # parent=int((ndx-1)/2)
            parent = int((ndx-1)/2)
            if self._elements[ndx] > self._elements[parent]:
                #                print(self._elements[ndx] , self._elements[parent])
                self._elements[ndx], self._elements[parent] = self._elements[parent], self._elements[ndx]
#                print(self._elements[ndx] , self._elements[parent])

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


def sort_heap(arr):
    mheap = MaxHeap(len(arr))

    for i in arr:
        mheap.add(i)
#    print(mheap)
    arr = [mheap.remove() for i in range(len(arr))]
    return arr


print(sort_heap([2, 3, 1, 5, 5, 1, 23, 6, 8, 12, 434]))
# 小顶堆实现递增排序


def sort_heap2(arr):
    import heapq
    heapq.heapify(arr)
    arr = [heapq.heappop(arr) for i in range(len(arr))]
    return arr


print(sort_heap2([2, 3, 1, 5, 5, 1, 23, 6, 8, 12, 434]))

"""
#计数排序
#计数排序的时间复杂度为 O(n + m )
    稳定排序
        有一个需求就是当对成绩进行排名次的时候，如何在原来排前面的人，
        排序后还是处于相同成绩的人的前面。
        解题的思路是对 countArr 计数数组进行一个变形，变来和名次挂钩，
        我们知道 countArr 存放的是分数的出现次数，那么其实我们可以算出每个分数的最大名次，
        就是将 countArr 中的每个元素顺序求和。
    计数排序只适用于正整数并且取值范围相差不大的数组排序使用，它的排序的速度是非常可观的。
"""


def sort_jishu(arr):  # O(n + m )
    # 最大元素
    max_i = max(arr)
    count_arr = [0]*(max_i+1)  # 计数数组
    # 计数
    for i in arr:
        count_arr[i] += 1
    # 顺序累加
    for i in range(1, len(count_arr)):
        count_arr[i] += count_arr[i-1]
    # 排序后的数组
    sorted_arr = [0]*len(arr)
    # 排序，从原数组从后往前遍历,此时计数排序稳定
    for i in reversed(arr):
        sorted_arr[count_arr[i]-1] = i
        count_arr[i] -= 1
    # sorted
    arr = sorted_arr
    return arr


print(sort_jishu([2, 3, 1, 5, 5, 1, 23, 6, 8, 12, 434]))

"""
    # 桶排序# O(n + m )
    要想保持稳定，即保持排名可以用和计数排序同样的方法
    在额外空间充足的情况下，尽量增大桶的数量，极限情况下每个桶只有一个数据时，
    或者是每只桶只装一个值时，完全避开了桶内排序的操作，桶排序的最好时间复杂度就能够达到 O(n)。
    # https://blog.csdn.net/qq_27124771/article/details/87651495
"""

def sort_bucket(arr):
    bucket_width = len(arr)  # 可以设置为常数，10，12...
    min_num = min(arr)  # 数组中的最小值
    bucket_num = int((max(arr)-min_num)/bucket_width+1)  # 桶的数量，+1是因为从0开始
    buckets = [[] for i in range(bucket_num)]  # 建立桶
    for i in arr:  # 将元素放入桶中
        bucket_idx = int((i-min_num)/bucket_width)
        buckets[bucket_idx].append(i)
    # 对每个桶进行排序
    for bucket in buckets:
        # 随便使用上面哪一种排序方法排序
        sort_charu(bucket)
    arr = [j for bucket in buckets for j in bucket]
    return arr


print(sort_bucket([2, 3, 1, 5, 5, 1, 23, 6, 8, 12, 434]))
"""
    基数排序(Radix Sort)是桶排序的扩展，它的基本思想是：
        将整数按位数切割成不同的数字，然后按每个位数分别比较。
        具体做法是：将所有待比较数值统一为同样的数位长度，数位较短的数前面补零。然后，从
        最低位开始，依次进行一次排序。这样从最低位排序一直到最高位排序完成以后, 数列就变成一个有序序列。
        由于整数也可以表达字符串（比如名字或日期）和特定格式的浮点数，所以基数排序也不是只能使用于整数。
        http://www.manongjc.com/detail/5-ekjkdmgbysuudff.html
"""
#基数排序
def sort_radix(arr):
    d=len(str(max(arr)))
    for i in range(d):#d轮排序
        s = [[] for k in range(10)]#因为每一位数字都是0~9，故建立10个桶
        for j in arr:
            s[int(j/(10**i)%10)].append(j)
        arr = [a for b in s for a in b]
    return arr
print(sort_radix([2, 3, 1, 5, 5, 1, 23, 6, 8, 12, 434]))
    
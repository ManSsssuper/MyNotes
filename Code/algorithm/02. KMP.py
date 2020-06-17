# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:42:00 2020

@author: ManSsssuper
        https://www.cnblogs.com/SYCstudio/p/7194315.html
        其实就是用了动态规划的思想，已经解决的问题要记住，无论是求F还是求index
"""
def get_f(pattern_str):
    F=[-1]*len(pattern_str)
    for i in range(1,len(F)):
        j=F[i-1]
        while pattern_str[j+1]!=pattern_str[i] and j>=0:
            j=F[j]
        if pattern_str[j+1]==pattern_str[i]:
            F[i]=j+1
    print('%s\n%s'%(pattern_str,F))
    return F

def get_idx(main_str,pattern_str):
    F=get_f(pattern_str)
    i=j=0
    while i<len(main_str):
        while i<len(main_str) and j<len(pattern_str) and main_str[i]==pattern_str[j]:
            i+=1
            j+=1
        
        if j==len(pattern_str):
            return i-j
        else:
            j=F[j-1]+1
    return None
print(get_idx('abaabaabbabaaabaabbabaab','abaabbabaab'))
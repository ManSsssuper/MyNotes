# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:08:50 2020

@author: ManSsssuper
"""

#normal/uniform正态分布/均匀分布用法???
import torch
print(torch.normal(torch.tensor([0]),torch.tensor([1])))

#torch.gather
print(torch.gather(x,0,torch.tensor([2,4])))

#torch.index_select
features.index_select(0, j), labels.index_select(0, j)

#
torch.manual_seed(1)

#assert
assert 0 <= drop_prob <= 1


idx = slice(j * fold_size, (j + 1) * fold_size)
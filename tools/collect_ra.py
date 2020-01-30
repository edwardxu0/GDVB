#!/usr/bin/env python
import os
import numpy as np

root = './results/'
seeds = [10,11,12,13,14]
name = 'dave' #'mcb'

ras = []

for s in seeds:
    path = root + f'{name}.{s}/dis_log/'

    files = os.listdir(path)
    
    ra10 = []
    
    for f in files:
        print(f'{path}/{f}')
        lines = [x.strip() for x in open(f'{path}/{f}','r').readlines() if 'validation error' in x]
        temp = [float(l.split('=')[3]) for l in lines]
        ra10 += [temp]
    ras += [ra10]

ras = np.array(ras)

print(ras.shape)
print(np.mean(ras,axis=1))
print(np.mean(np.mean(ras,axis=1),axis=0))

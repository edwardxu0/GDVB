#!/usr/bin/env python
import os
import sys


tmp_file = './tmp/tmp.txt'

dis_dir = sys.argv[1]
cmd = f'ls -lovh {dis_dir} > {tmp_file}'
os.system(cmd)

lines = [x for x in open(tmp_file, 'r').readlines() if '.out' in x]

for l in lines:
    tks = l.strip().split(' ')
    if 'mnist' in dis_dir:
        tag = '2.2M'
    elif 'cifar' in dis_dir:
        tag = '1.9M'
    else:
        assert False

    if tks[3] != tag:
        cmd = f'rm  {dis_dir}/{tks[-1]}'
        print(cmd)
        os.system(cmd)
        cmd = f'rm  {dis_dir}/{tks[-1]}'[:-3]+'err'
        print(cmd)
        os.system(cmd)

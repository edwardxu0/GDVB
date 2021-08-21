#!/usr/bin/env python
import os


tmp_file = 'tmp/jobs.txt'
cmd = f'squeue -u dx3yy > {tmp_file}'
os.system(cmd)
jobs = [x for x in open(tmp_file,'r').readlines() if 'JobHoldMaxRequeue' in x]
for j in jobs:
    tks = j.strip().split(' ')
    idaa = int(tks[0])
    cmd = f'scancel {idaa}'
    print(cmd)
    os.system(cmd)

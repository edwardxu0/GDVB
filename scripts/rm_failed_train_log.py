#!/usr/bin/env python
import os
import sys


tmp_file = './tmp/tmp.txt'

if len(sys.argv) < 3:
    print(f'{sys.argv[0]} [log_dir] [retain TAG]')
    exit()

dis_dir = sys.argv[1]
tag = sys.argv[2]

cmd = f'ls -lovh {dis_dir} > {tmp_file}'
os.system(cmd)

lines = [x for x in open(tmp_file, 'r').readlines() if '.out' in x]

junk = []

for l in lines:
    tks = l.strip().split(' ')
    print(tks[3])
    if tks[3] != tag:
        junk += [f'{dis_dir}/{tks[-1]}']
        junk += [f'{dis_dir}/{tks[-1]}'[:-3]+'err']

print(f'Failed training jobs: {len(junk)/2}')

#for j in junk:
#    os.remove(j)

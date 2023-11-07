#!/usr/bin/env python
import os
import sys


tmp_file = './tmp/tmp.txt'

if len(sys.argv) < 4:
    print(f'{sys.argv[0]} [log_dir] [retain TAG] [REMOVE(0/1)]')
if len(sys.argv) < 3:
    exit()

dis_dir = sys.argv[1]
tag = sys.argv[2]

cmd = f'ls -lovh {dis_dir} > {tmp_file}'
os.system(cmd)

lines = [x for x in open(tmp_file, 'r').readlines() if '.out' in x]

junk = []

for l in lines:
    tks = l.strip().split(' ')#
    #if tks[3] != tag:#
    if tag not in l:
        junk += [f'{dis_dir}/{tks[-1]}']
        junk += [f'{dis_dir}/{tks[-1]}'[:-3]+'err']

print(f'Failed training jobs: {int(len(junk)/2)}')

if len(sys.argv) > 3 and int(sys.argv[3]) == 1:
    for j in junk:
        print(f'removing {j}')
        os.remove(j)

    print('Failed logs removed.')
else:
    print('Use Flag {1} to remove failed logs.')

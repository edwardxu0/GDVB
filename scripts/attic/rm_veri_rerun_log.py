#!/usr/bin/env python
import os
import sys

lines = open(sys.argv[1], 'r').readlines()

c = 0
for l in lines:
    if l.startswith('rerun') or l.startswith('undetermined'):
        c+=1
        path = l.strip().split(' ')[1]
        cmd = f'rm {path}'
        print(cmd)
        os.system(cmd)
print(c)

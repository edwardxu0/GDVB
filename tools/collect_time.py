#!/usr/bin/env python
import os
import numpy as np
import sys
import datetime

root = './results/'
seeds = [10,11,12,13,14]
name = sys.argv[1]


def calc_time(log_path):
    ras = []

    for s in seeds:
        path = root + f'{name}.{s}/{log_path}/'

        files = os.listdir(path)

        ra10 = []
        for f in files:
            #print(f'{path}/{f}')

            lines = [x.strip() for x in open(f'{path}/{f}','r').readlines() if 'DEBUG    20' in x or 'INFO     20' in x]
            if lines[0].startswith('DEBUG'):
                date_s = [int(x) for x  in lines[0].split(' ')[4].split('-')]
                time_s = [int(float(x)) for x  in lines[0].split(' ')[5][:-4].split(':')]
            elif lines[0].startswith('INFO'):
                date_s = [int(x) for x  in lines[0].split(' ')[5].split('-')]
                time_s = [int(float(x)) for x  in lines[0].split(' ')[6][:-4].split(':')]
            else:
                assert False
            if lines[-1].startswith('DEBUG'):
                date_e = [int(x) for x  in lines[-1].split(' ')[4].split('-')]
                time_e = [int(float(x)) for x  in lines[-1].split(' ')[5][:-4].split(':')]
            elif lines[-1].startswith('INFO'):
                date_e = [int(x) for x  in lines[-1].split(' ')[5].split('-')]
                time_e = [int(float(x)) for x  in lines[-1].split(' ')[6][:-4].split(':')]
            start = datetime.datetime(*(date_s+time_s))
            end = datetime.datetime(*(date_e+time_e))
            delta = (end-start).total_seconds()/60/60

            ra10 += [delta]
        ras += [ra10]

    ras = np.array(ras)

    print(ras.shape)
    print('mean(h): ',np.mean(ras,axis=1),',average(h):',np.mean(ras))
    print('max(h): ',np.max(ras,axis=1),',average(h):',np.mean(np.max(ras,axis=1)))
    print('sum(h): ',np.sum(ras,axis=1),',average(h):',np.mean(np.sum(ras,axis=1)))

calc_time('dis_log')
calc_time('veri_log')

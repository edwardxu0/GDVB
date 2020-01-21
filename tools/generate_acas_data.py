#!/usr/bin/env python
import os
import argparse
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description='generate data for the ACAS network.')
    parser.add_argument('--scale', type=float, default=1, help='scale the range of the data')
    parser.add_argument('--size', type=int, default=100000, help='size of the data')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--root', type=str, default='./data/acas', help='')
    return parser.parse_args()


def generate_acas_data(scale, size):
    x_min = np.array([0.0,-3.141593,-3.141593,100.0,0.0,])
    x_max = np.array([60760.0,3.141593,3.141593,1200.0,1200.0,])
    x_mean = np.array([1.9791091e+04,0.0,0.0,650.0,600.0,])
    x_range = np.array([60261.0,6.28318530718,6.28318530718,1100.0,1200.0,])

    x_min_norm = (x_min - x_mean)/x_range
    x_max_norm = (x_max - x_mean)/x_range
    
    x_mean_norm = np.mean(np.array([x_min_norm, x_max_norm]), axis=0)
    x_min_norm = x_mean_norm-(x_mean_norm - x_min_norm) * scale
    x_max_norm = x_mean_norm+(x_max_norm - x_mean_norm) * scale

    data = []
    for i in range(len(x_min)):
        data+=[np.random.uniform(low = x_min_norm[i], high=x_max_norm[i], size=size)]
    data = np.array(data).T
    return data


if __name__ == '__main__':
    args = _parse_args()
    np.random.seed(args.seed)
    if not os.path.exists(args.root):
        os.mkdir(args.root)
        
    data = generate_acas_data(args.scale, args.size)
    np.save(os.path.join(args.root,f'acas.train.{args.scale}'), data)

    data = generate_acas_data(args.scale, int(args.size/10))
    np.save(os.path.join(args.root,f'acas.valid.{args.scale}'), data)

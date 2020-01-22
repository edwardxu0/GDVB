#!/usr/bin/env python
import os
import pathlib
import argparse
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description='generate data for the ACAS network.')
    parser.add_argument('--scale', type=float, default=1, help='scale the range of the data')
    parser.add_argument('--output_dir', type=str, default='./props/acas/', help='')
    
    return parser.parse_args()


def prop_skeleton(scale):
    return [
        "from dnnv.properties import *",
        "import numpy as np",
        "N = Network('N')",
        "x_min = np.array([0.0,-3.141593,-3.141593,100.0,0.0,])",
        "x_max = np.array([60760.0,3.141593,3.141593,1200.0,1200.0,])",
        "x_mean = np.array([1.9791091e+04,0.0,0.0,650.0,600.0,])",
        "x_range = np.array([60261.0,6.28318530718,6.28318530718,1100.0,1200.0,])",
        "x_min = (x_min-x_mean)/x_range * {scale}",
        "x_max = (x_max-x_mean)/x_range * {scale}",
        "y_mean = 7.5188840201005975",
        "y_range = 373.94992"]


def p1(scale):
    prop_lines = prop_skeleton(scale)[:7]
    prop_lines +=[
        "x_min[0] = 55947.691",
        "x_min[3] = 11",
        "x_max[4] = 60"
        ]
    prop_lines += prop_skeleton(scale)[7:]
    prop_lines +=[
        "Forall(",
        "    x,",
        "    Implies(",
        "        (x_min < x < x_max),",
        "        N(x)[0] <= (1500 - y_mean)/y_range",
        "    ),",
        ")"
    ]
    prop_lines = [x+'\n' for x  in prop_lines]
    return prop_lines


def p3(scale):
    prop_lines = prop_skeleton(scale)[:7]
    prop_lines +=[
        "x_min[0] = 1500",
        "x_max[0] = 1800",
        "x_min[1] = -0.06",
        "x_max[1] = 0.06",
        "x_min[2] = 3.10",
        "x_min[3] = 980",
        "x_min[4] = 960"]
    prop_lines += prop_skeleton(scale)[7:]
    prop_lines +=[
        "Forall(",
        "    x,",
        "    Implies(",
        "        (x_min < x < x_max),",
        "        Or(N(x)[0]>N(x)[1], N(x)[0]>N(x)[2], N(x)[0]>N(x)[3], N(x)[0]>N(x)[4])",
        "    ),",
        ")"]
    prop_lines = [x+'\n' for x  in prop_lines]
    return prop_lines


def p4(scale):
    prop_lines = prop_skeleton(scale)[:7]
    prop_lines +=[
        "x_min[0] = 1500",
        "x_max[0] = 1800",
        "x_min[1] = -0.06",
        "x_max[1] = 0.06",
        "x_min[2] = 0",
        "x_max[2] = 0",
        "x_min[3] = 1000",
        "x_min[4] = 700",
        "x_max[4] = 800"]
    prop_lines += prop_skeleton(scale)[7:]
    prop_lines +=[
        "Forall(",
        "    x,",
        "    Implies(",
        "        (x_min < x < x_max),",
        "        Or(N(x)[0]>N(x)[1], N(x)[0]>N(x)[2], N(x)[0]>N(x)[3], N(x)[0]>N(x)[4])",
        "    ),",
        ")"]
    prop_lines = [x+'\n' for x  in prop_lines]    
    return prop_lines


def p5(scale):
    prop_lines = prop_skeleton(scale)[:7]
    prop_lines +=[
        "x_min[0] = 250",
        "x_max[0] = 400",
        "x_min[1] = 0.2",
        "x_max[1] = 0.4",
        "x_min[2] = -3.141592",
        "x_max[2] = -3.141592+0.005",
        "x_min[3] = 100",
        "x_max[3] = 400",
        "x_min[4] = 0",
        "x_max[4] = 400"]

    prop_lines += prop_skeleton(scale)[7:]
    prop_lines +=[
        "Forall(",
        "    x,",
        "    Implies(",
        "        (x_min < x < x_max),",
        "        argmin(N(x)) == 4,",
        "    ),",
        ")"]
    prop_lines = [x+'\n' for x  in prop_lines]
    return prop_lines

    
def p6(scale):
    prop_lines = prop_skeleton(scale)[:7]
    prop_lines +=[
        "x_min[0] = 12000",
        "x_max[0] = 62000",
        "x_min[1] = np.array([0.7,-3.141592])",
        "x_max[1] = np.array([3.141592,-0.7])",
        "x_min[2] = -3.141592",
        "x_max[2] = -3.141592+0.005",
        "x_min[3] = 100",
        "x_max[3] = 1200",
        "x_min[4] = 0",
        "x_max[4] = 1200"]
    prop_lines += prop_skeleton(scale)[7:]
    prop_lines +=[
        "Forall(",
        "    x,",
        "    Implies(",
        "        And(",
        "            x_min[0] <= x[0] <= x_max[0],",
        "            Or(x_min[1][0] <= x[1] <= x_max[1][0], x_min[1][1] <= x[1] <= x_max[1][1]),",
        "            x_min[2] <= x[2] <= x_max[2],",
        "            x_max[3] <= x[3] <= x_max[3],",
        "            x_max[4] <= x[4] <= x_max[4]",
        "        ),",
        "        argmin(N(x)) == 0",
        "    )",
        ")"]
    prop_lines = [x+'\n' for x  in prop_lines]
    return prop_lines


if __name__ == '__main__':
    args = _parse_args()    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    properties = [1,3,4,5,6]
    properties = {x:globals()[f'p{x}'](args.scale) for x in properties}

    for k in properties:
        open(os.path.join(args.output_dir, f'{k}.{args.scale}.py'),'w').writelines(properties[k])

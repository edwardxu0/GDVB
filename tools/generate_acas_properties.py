#!/usr/bin/env python
import os
import pathlib
import argparse
import numpy as np
import copy


def _parse_args():
    parser = argparse.ArgumentParser(description='generate data for the ACAS network.')
    parser.add_argument('-s','--scale', type=float, default=1, help='scale the range of the data')
    parser.add_argument('-d','--output_dir', type=str, default='./props/acas/', help='')
    
    return parser.parse_args()


def prop_skeleton(mmmr,scale):
    return [
        "from dnnv.properties import *",
        "import numpy as np",
        "N = Network('N')",
        f"x_min = np.array([{mmmr[0]}])",
        f"x_max = np.array([{mmmr[1]}])",
        f"x_mean = np.array([{mmmr[2]}])",
        f"x_range = np.array([{mmmr[3]}])",
        f"x_min = (x_min-x_mean)/x_range * {scale}",
        f"x_max = (x_max-x_mean)/x_range * {scale}",
        "y_mean = 7.5188840201005975",
        "y_range = 373.94992"]


def p1(mmmr,scale):
    mmmr[0][0] = 55947.691
    mmmr[0][3] = 11
    mmmr[1][4] = 60
    
    prop_lines = prop_skeleton(mmmr,scale)
    prop_lines +=[
        "Forall(",
        "    x,",
        "    Implies(",
        "        (x_min < x < x_max),",
        "        N(x)[0,0] <= (1500 - y_mean)/y_range",
        "    ),",
        ")"
    ]
    prop_lines = [x+'\n' for x  in prop_lines]
    return prop_lines


def p3(mmmr,scale):
    mmmr[0][0] = 1500
    mmmr[1][0] = 1500
    mmmr[0][1] = -0.06
    mmmr[1][1] = 0.06
    mmmr[0][2] = 3.10
    mmmr[0][3] = 980
    mmmr[0][4] = 960
    
    prop_lines = prop_skeleton(mmmr,scale)
    prop_lines +=[
        "Forall(",
        "    x,",
        "    Implies(",
        "        (x_min < x < x_max),",
        "        Or(N(x)[0,0]>N(x)[0,1], N(x)[0,0]>N(x)[0,2], N(x)[0,0]>N(x)[0,3], N(x)[0,0]>N(x)[0,4])",
        "    ),",
        ")"]
    prop_lines = [x+'\n' for x  in prop_lines]
    return prop_lines


def p4(mmmr,scale):
    mmmr[0][0] = 1500
    mmmr[1][0] = 1800
    mmmr[0][1] = -0.06
    mmmr[1][1] = 0.06
    mmmr[0][2] = 0
    mmmr[1][2] = 0
    mmmr[0][3] = 1000
    mmmr[0][4] = 700
    mmmr[1][4] = 800
    
    prop_lines = prop_skeleton(mmmr,scale)
    prop_lines +=[
        "Forall(",
        "    x,",
        "    Implies(",
        "        (x_min < x < x_max),",
        "        Or(N(x)[0,0]>N(x)[0,1], N(x)[0,0]>N(x)[0,2], N(x)[0,0]>N(x)[0,3], N(x)[0,0]>N(x)[0,4])",
        "    ),",
        ")"]
    prop_lines = [x+'\n' for x  in prop_lines]    
    return prop_lines


def p5(mmmr,scale):
    mmmr[0][0] = 250
    mmmr[1][0] = 400
    mmmr[0][1] = 0.2
    mmmr[1][1] = 0.4
    mmmr[0][2] = -3.141592
    mmmr[1][2] = -3.141592+0.005
    mmmr[0][3] = 100
    mmmr[1][3] = 400
    mmmr[0][4] = 0
    mmmr[1][4] = 400
    
    prop_lines = prop_skeleton(mmmr,scale)
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

    
def p6(mmmr,scale):
    mmmr[0][0] = 12000
    mmmr[1][0] = 62000
    mmmr[0][1] = 0.7
    mmmr[1][1] = 3.141592
    mmmr[0][2] = -3.141592
    mmmr[1][2] = -3.141592+0.005
    mmmr[0][3] = 100
    mmmr[1][3] = 1200
    mmmr[0][4] = 0
    mmmr[1][4] = 1200

    x_min2 = copy.deepcopy(mmmr[0])
    x_max2 = copy.deepcopy(mmmr[1])
    x_min2[1] = -3.141592
    x_max2[1] = -0.7
    
    prop_lines = prop_skeleton(mmmr,scale)
    prop_lines +=[
        f"x_min2 = np.array([{x_min2}])",
        f"x_max2 = np.array([{x_max2}])",
        f"x_min2 = (x_min2-x_mean)/x_range * {scale}",
        f"x_max2 = (x_max2-x_mean)/x_range * {scale}",
        "Forall(",
        "    x,",
        "    Implies(",
        "        Or(x_min < x < x_max, x_min2 < x < x_max2),",
        "        argmin(N(x)) == 0",
        "    )",
        ")"]
    
    prop_lines = [x+'\n' for x  in prop_lines]
    return prop_lines


if __name__ == '__main__':
    args = _parse_args()    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    x_min = [0.0,-3.141593,-3.141593,100.0,0.0]
    x_max = [60760.0,3.141593,3.141593,1200.0,1200.0]
    x_mean = [19791.091,0.0,0.0,650.0,600.0]
    x_range = [60261.0,6.28318530718,6.28318530718,1100.0,1200.0]
    mmmr = [x_min,x_max,x_mean,x_range]

    properties = [1,3,4,5,6]
    properties = {x:globals()[f'p{x}'](copy.deepcopy(mmmr), args.scale) for x in properties}

    for k in properties:
        open(os.path.join(args.output_dir, f'{k}_{args.scale}.py'),'w').writelines(properties[k])

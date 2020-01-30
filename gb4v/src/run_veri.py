#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import random
import skimage
import math


random.seed(0)
np.random.seed(0)

ORIG_IMG_SIZES = {"imagenet":224,
                  "driving":100
}
DATASET_OF = { "vgg":"imagenet",
               "dave":"driving",
               "resset":"imagenet"
}
TOOL_CMD = { "eran": "python cegsdl/tools/runERAN.py"
}

SLURM_FILE = "run.slurm"
SLURM_LOG_DIR = "slurmlogs"


def _parse_args():
    parser = argparse.ArgumentParser(description="Run verification.")
    parser.add_argument("verifier", type=str, choices=["reluplex", "planet", "eran", "mipverify", "cnncert"])
    parser.add_argument("network", type=str, choices=["vgg","dave","resnet"])
    parser.add_argument("epsilon",type=float)
    return parser.parse_args()

def write_slrum_file(verifier, network, model_path, prop_path, epsilon, true_label):
    if not os.path.exists(SLURM_LOG_DIR):
        os.mkdir(SLURM_LOG_DIR)

    lines = ["#!/bin/sh"]
    lines += ["#SBATCH --time=1000:00:00"]
    lines += ["#SBATCH --mem=2"]
    lines += ["#SBATCH --job-name={}-{}".format(verifier, network)]
    lines += ['#SBATCH --output="{}/{}_{}_{}_{}_{}_{}.out"'.format(SLURM_LOG_DIR,
                                                                  verifier,
                                                                  network,
                                                                  model_path,
                                                                  prop_path,
                                                                  epsilon,
                                                                  true_label
                                                              )]
    lines += ['#SBATCH --error="{}/{}_{}_{}_{}_{}_{}.out"'.format(SLURM_LOG_DIR,
                                                                  verifier,
                                                                  network,
                                                                  model_path,
                                                                  prop_path,
                                                                  epsilon,
                                                                  true_label
                                                              )]
    lines += [""]
    lines += ["$@"]
    lines = [l+'\n' for l in lines]
    
    slurm_file = open(SLURM_FILE,'w')
    slurm_file.writelines(lines)
    

def run_all(args):
    verifier = args.verifier
    network = args.network
    model_dir = args.model_dir
    prop_dir = args.prop_dir
    epsilon = args.epsilon
    
    dataset = DATASET_OF[network]
    
    models = os.listdir(model_dir)

    if verifier == 'reluplex':
    for m in models:
        model_path = os.path.join(model_dir, m)
        tokens=m.split('.')
        if len(tokens)<3:
            input_resize_factor = 1
        else:
            input_resize_factor = tokens[2]
            input_resize_factor = float(input_resize_factor)/(math.pow(10,len(input_resize_factor)))
        img_size = int(ORIG_IMG_SIZES[dataset] * input_resize_factor)
            
        
        props = os.listdir(os.path.join(prop_dir, dataset+'.'+str(img_size)))

        props = props[:1]
        for p in props:
            prop_path = os.path.join(prop_dir, dataset+'.'+str(img_size), p)
            true_label = p.split('_')[0]

            write_slrum_file(verifier, network, m, p, epsilon, true_label)
            task = "sbatch -w trillian2 {} {} {} {} {} {} --input_shape 1 3 {} {}".format(SLURM_FILE, TOOL_CMD[verifier], model_path, prop_path, epsilon, true_label, img_size, img_size)
            print(task)
            os.system(task)
            
            '''
            task = "{} {} {} {} {} --input_shape 1 3 {} {}".format(TOOL_CMD[verifier], model_path, prop_path, epsilon, true_label, img_size, img_size)
            print(task)
            os.system(task)
            '''
    

if __name__ == "__main__":
    run_all(_parse_args())

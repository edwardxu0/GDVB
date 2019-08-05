#!/usr/bin/env python
import os
import argparse
import random
import numpy as np
import copy
import toml
import time
from itertools import combinations

from nn.onnxu import Onnxu

from network import Network


SEED = 0
random.seed(SEED)
np.random.seed(SEED)

CONFIG_DIR = 'res_config'
DIS_DIR = 'res_dis'
VERI_DIR = 'res_veri'

VERIFIERS = ['neurify']
PROPERTIES_DIR = 'properties'

def _parse_args():
    parser = argparse.ArgumentParser(description='Neural Architecture Search')
    parser.add_argument('-om', '--original_model', type=str, help='the original onnx model', default = '/p/d4v/dx3yy/DNNVeri/dnnvbs/cegsdl/networks/dave/model.onnx')
    parser.add_argument('-dc', '--distillation_configuration', type=str, help='the general configuration file for distillation', default='/p/d4v/dx3yy/DNNVeri/dnnvbs/configs/dave.toml')
    parser.add_argument('--network_name', type=str, default='dave')
    return parser.parse_args()


CONFIGS={
    'root':'/p/d4v/dx3yy/nas_bad_res/',
    'proxy_dis_epochs':12,
    'scale_layer_factor':.5,
    'dis_threshold':{'dave':['loss',0.1,0.1]},
    'veri_time': 60, # verification time in seconds
    'veri_mem': 64, # verification memory in Gigabytes
    'proxy_nb_props':5,
    'veri_threshold':[1,1],
    'verifiers': ['neurify']}


def configure(args):
    if not os.path.exists(CONFIGS['root']):
        os.mkdir(CONFIGS['root'])

    subdirs = ['dis_config_dir','dis_model_dir','dis_slurm_dir','dis_log_dir','dis_done_dir','test_slurm_dir','test_log_dir','test_done_dir','prop_dir','veri_net_dir','veri_slurm_dir','veri_log_dir','veri_done_dir']

    for sd in subdirs:
        CONFIGS[sd] = os.path.join(CONFIGS['root'],sd[:-4])
        if not os.path.exists(CONFIGS[sd]):
            os.mkdir(CONFIGS[sd])

    CONFIGS['network_name'] = args.network_name


def get_layers(args, onnx_net):
    layers_with_ids = ['Conv', 'FC', 'Transpose', 'Flatten']
    layers = []
    for l in onnx_net.arch:
        if l.type in layers_with_ids:
            layers += [l]
    if args.network_name == 'dave':
        layers = layers[1:]
    else:
        assert False
    return layers         


def NAS_BS(args):
    onnx_net = Onnxu(args.original_model)
    droppable_scalable_layers = get_layers(args, onnx_net)

    # 1. binary search phase 1
    transformations_drop = ['D0', 'D1', 'D2', 'D3', 'D4', 'D7', 'D8', 'D9', 'D10']

    comb_trans_drop = []
    for i in range(len(transformations_drop)):
        comb_trans_drop += combinations(transformations_drop, i+1)

    distillation_config = open(args.distillation_configuration,'r').read()
    distillation_config = toml.loads(distillation_config)

    #original_network = Network(args.network_name, CONFIGS)

    networks = []

    for ctd in comb_trans_drop:
        network = Network(args.network_name, CONFIGS)
        network.set_distillation_strategies(distillation_config, ctd)
        network.calc_order('nb_neurons', droppable_scalable_layers)
        
        # TODO: make this general
        if np.sum(network.nb_neurons) < 82669:
            networks += [network]


    networks = sorted(networks)

    bad_search(networks)


def bad_search(networks):

    networks = networks[14:]
    c = 0
    for n in networks:
        if not os.path.exists(n.dis_done_path):
            c +=1
            n.distill()
    print(c)

    
def binary_search(networks, L, R, phase, d=0):
    if L>R:
        return
    M = (L+R)//2
    net = networks[M]

    print('\n---- INFO ---- Start on BS network:', net.name, net.distilled)

    #distill
    net.distill()

    #test
    net.test()
    net.analyze_test_results()

    #transform
    net.transform()

    #verify
    net.verify()
    net.analyze_veri_results()
    

    net.score(1)
    print('Net score', net.score, net.score_ra, net.score_veri)

    if L < M:
        if L+1==M and networks[L].distilled:
            return
        else:
            l_best = binary_search(networks, L, M, d+1)
        
    if M < R:
        if M+1==R and networks[M].distilled:
            return
        else:
            r_best = binary_search(networks, M, R, d+1)


def main(args):
    configure(args)
    NAS_BS(args)


if __name__ == '__main__':
    main(_parse_args())

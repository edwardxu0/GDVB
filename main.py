#!/usr/bin/env python
import os
import argparse
import random
import numpy as np
import copy
import toml
import time
import json
from itertools import combinations

from nn.onnxu import Onnxu

from network import Network


CONFIG_DIR = 'res_config'
DIS_DIR = 'res_dis'
VERI_DIR = 'res_veri'

VERIFIERS = ['neurify']
PROPERTIES_DIR = 'properties'

def _parse_args():
    parser = argparse.ArgumentParser(description='Neural Architecture Search')
    parser.add_argument('config', type=str, help='NAS config')
    return parser.parse_args()


def configure(args):

    global CONFIGS
    '''
    CONFIGS={
    'root':'/p/d4v/dx3yy/nas_bad_res/',
    'nb_best_net':5,
    'proxy_dis_epochs':12,
    'scale_layer_factor':.5,
        'dis_threshold':{'dave':{'type':'loss','value':[0.1,0.1]}},
    'veri_time': 60, # verification time in seconds
    'veri_mem': 64, # verification memory in Gigabytes
    'proxy_nb_props':5,
    'veri_threshold':[1,1],
    'verifiers': ['neurify']}

    configs = toml.dumps(CONFIGS)
    with open('temp.toml','w') as f:
        f.write(configs)
    '''

    config_file = open(args.config,'r').read()
    CONFIGS = toml.loads(config_file)

    if not os.path.exists(CONFIGS['root']):
        os.mkdir(CONFIGS['root'])

    subdirs = ['dis_config_dir','dis_model_dir','dis_slurm_dir','dis_log_dir','dis_done_dir','test_slurm_dir','test_log_dir','test_done_dir','prop_dir','veri_net_dir','veri_slurm_dir','veri_log_dir','veri_done_dir']

    for sd in subdirs:
        CONFIGS[sd] = os.path.join(CONFIGS['root'],sd[:-4])
        if not os.path.exists(CONFIGS[sd]):
            os.mkdir(CONFIGS[sd])


def get_layers(onnx_net):
    layers_with_ids = ['Conv', 'FC', 'Transpose', 'Flatten']
    layers = []
    for l in onnx_net.arch:
        if l.type in layers_with_ids:
            layers += [l]
    if CONFIGS['network_name'] == 'dave':
        layers = layers[1:]
    else:
        assert False
    return layers         


def NAS_BS():
    onnx_net = Onnxu(CONFIGS['original_model'])
    droppable_scalable_layers = get_layers(onnx_net)

    # 1. binary search phase 1
    transformations_drop = ['D0', 'D1', 'D2', 'D3', 'D4', 'D7', 'D8', 'D9', 'D10']

    comb_trans_drop = []
    for i in range(len(transformations_drop)):
        comb_trans_drop += combinations(transformations_drop, i+1)

    #original_network = Network(args.network_name, CONFIGS)

    networks = []

    for ctd in comb_trans_drop:
        network = Network(CONFIGS)
        network.set_distillation_strategies(ctd)
        network.calc_order('nb_neurons', droppable_scalable_layers)
    
        # TODO: make this general
        if np.sum(network.nb_neurons) < 82669:
            networks += [network]

    networks = sorted(networks)

    if CONFIGS['mode'] == 'nas_bs':
        binary_search(networks, 0, len(networks)-1, phase=1)
    elif CONFIGS['mode'] == 'nas_bs_simu':
        nas_bs_simu(networks)
    else:
        assert False

    if CONFIGS['dis_threshold'][CONFIGS['network_name']]['type'] == 'loss':
        networks = sorted(networks, key=lambda x:(x.score_veri, -x.score_ra), reverse=True)
    elif CONFIGS['dis_threshold'][CONFIGS['network_name']]['type'] == 'acc':
        networks = sorted(networks, key=lambda x:(x.score_veri, x.score_ra), reverse=True)
    else:
        assert False

    best_networks = networks[:CONFIGS['nb_best_net']]
    print(len(best_networks))
    
    # 2. binary search phase 2
    networks2 = []
    transformations_scale = ['S0', 'S1', 'S2', 'S3', 'S4', 'S7', 'S8', 'S9', 'S10']

    comb_trans_scale = []
    for i in range(len(transformations_scale)):
        comb_trans_scale += combinations(transformations_scale, i+1)
    
    for cts in comb_trans_scale:
        for bn in best_networks:
            scale_ids = [int(x[1:]) for x in cts]
            dis_strats = list(bn.dis_strategies) + list(cts)

            if not list(set(bn.drop_ids).intersection(scale_ids)):
                n = Network(CONFIGS)
                n.set_distillation_strategies(dis_strats)
                n.calc_order('nb_neurons', droppable_scalable_layers)

                # TODO: make this general
                if np.sum(n.nb_neurons) < 82669:
                    networks2 += [n]

    print(len(networks2))
    print([x.dis_strategies for x in networks2])
    

    networks2 = sorted(networks2)

    if CONFIGS['mode'] == 'nas_bs':
        binary_search(networks2, 0, len(networks)-1, phase=2)
    elif CONFIGS['mode'] == 'nas_bs_simu':
        nas_bs_simu(networks2)
    else:
        assert False


def nas_bs_simu(networks):

    nets = []

    for n in networks:
        if not os.path.exists(n.dis_model_dir):
            nets += [n]
        else:
            if not os.path.exists(os.path.join(n.dis_model_dir,n.name+'.iter.10.onnx')):
                nets += [n]
    
    print(len(nets))
    for n in nets:
        n.distill()

    for n in networks:
        if not os.path.exists(n.test_done_path):
            nets += [n]

    nets = nets[24:]
    print(len(nets))
    exit()
    for n in nets:
        n.test()

        

    
def binary_search(networks, L, R, phase, d=0):
    if L>R:
        return
    elif L==R and networks[L].processed:
        return

    M = (L+R)//2
    net = networks[M]
    if net.processed:
        return

    print('\n---- INFO ---- Start on BS network:', net.name, net.distilled, np.sum(net.nb_neurons))

    #distill
    net.distill()
    net.dis_monitor()

    #test
    net.test()
    net.test_monitor()

    #transform
    net.transform()

    #verify
    net.verify()
    net.veri_monitor()
    time.sleep(1)
    net.score()
    net.processed = True

    if phase == 1:
        print('Net score', net.score_ra, net.accurate(1), net.score_veri, net.verifiable(1))
        if not net.verifiable(1):
            binary_search(networks, L, M, d+1)
        else:
            binary_search(networks, M, R, d+1)

    elif phase == 2:
        print('Net score', net.score, net.score_ra, net.accurate(1), net.score_veri, net.verifiable(1))
        if net.accurate(2) and not net.verifiable(2):
            binary_search(networks, L, M, d+1)
        elif not net.accurate(2) and net.verifiable(2):
            binary_search(networks, M, R, d+1)
        elif not net.accurate(2) and not net.verifiable(2):
            binary_search(networks, L, M, d+1)
            binary_search(networks, M, R, d+1)
    else:
        assert False


def main(args):
    configure(args)
    NAS_BS()


if __name__ == '__main__':
    main(_parse_args())

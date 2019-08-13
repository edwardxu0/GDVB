#!/usr/bin/env python
import os
import sys
import argparse
import random
import numpy as np
import copy
import toml
import time
import logging
import threading
import json
from itertools import combinations

logging.basicConfig(stream=sys.stdout,
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

from nn.onnxu import Onnxu

from network import Network


CONFIG_DIR = 'res_config'
DIS_DIR = 'res_dis'
VERI_DIR = 'res_veri'

VERIFIERS = ['neurify']
PROPERTIES_DIR = 'properties'


def _parse_args():
    parser = argparse.ArgumentParser(description='Neural Architecture Search')
    parser.add_argument('alg', type=str, help='NAS Algorithm')
    parser.add_argument('config', type=str, help='NAS config')
    parser.add_argument('scale_phases', type=int, help='# of scale phases')
    
    return parser.parse_args()


def configure(args,p=0):
    config_file = open(args.config,'r').read()
    global CONFIGS
    CONFIGS = toml.loads(config_file)
    if p == 1:
        CONFIGS['root'] += '_al'

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

def calc_neurons(onnx_path):
    onnx_net = Onnxu(onnx_path)
    droppable_scalable_layers = get_layers(onnx_net)
    n = Network(CONFIGS)
    n.set_distillation_strategies([])
    n.calc_order('nb_neurons', droppable_scalable_layers)
    print(np.sum(n.nb_neurons))


def NAS_BS(args):
    onnx_net = Onnxu(CONFIGS['original_model'])
    droppable_scalable_layers = get_layers(onnx_net)

    # 1. binary search phase 1: drop only
    logging.info('Phase 1 started.')
    orig_net_size = 82669
    trans = [0,1,2,3,4,7,8,9,10]
    trans_comb = []
    for i in range(len(trans)):
        trans_comb += combinations(trans, i+1)
    trans_comb = [list(x) for x in trans_comb]

    nets = []
    for t in trans_comb:
        n = Network(CONFIGS)
        dis_strats = [['drop',x ,1] for x in t]
        n.set_distillation_strategies(dis_strats)
        n.calc_order('nb_neurons', droppable_scalable_layers)
        if np.sum(n.nb_neurons) < orig_net_size:
            nets += [n]

    nets = sorted(nets)
    binary_search(nets, 0, len(nets)-1, phase=1)
    logging.info('Phase 1 finished.')
    plot_data(nets, 1, args.config.split('/')[1][:-5]+'.txt')

    all_nets = [nets]
    for p in range(args.scale_phases):
        logging.info('Phase {} started.'.format(p+2))
        # promot top networks as seed networks to the next phase
        nets_seed = all_nets[-1]
        if CONFIGS['dis_threshold'][CONFIGS['network_name']]['type'] == 'loss':
            nets_seed = sorted(nets_seed, key=lambda x:(x.score_veri, -x.score_ra), reverse=True)
        elif CONFIGS['dis_threshold'][CONFIGS['network_name']]['type'] == 'acc':
            nets_seed = sorted(nets_seed, key=lambda x:(x.score_veri, x.score_ra), reverse=True)
        else:
            assert False
        nets_seed = nets_seed[:CONFIGS['nb_best_net']]

        nets_threads = []
        threads = []
        for ns in nets_seed:
            trans = [0,1,2,3,4,7,8,9,10]
            comb_trans = []
            max_scale = CONFIGS['max_nb_layer_scale'] if CONFIGS['scale_limit_strat'] == 'nb_layers' else len(trans)
            for i in range(max_scale):
                comb_trans += combinations(trans, i+1)
            comb_trans = [list(x) for x in comb_trans]

            nets = []
            for cts in comb_trans:
                dis_strats  = []
                # For a scaled network if verifiable, scale up; otherwise, scale down
                scale_factor = 1+0.5 if ns.verifiable(1) else 1-0.5
                for sds in ns.dis_strats:
                    if sds[1] in cts:
                        if sds[0] == 'drop':
                            dis_strats += [['scale',sds[1],(1-0.5)]]
                        elif sds[0] == 'scale':
                            dis_strats += [['scale',sds[1],sds[2]*scale_factor]]
                        else:
                            assert False
                        cts.remove(sds[1])
                    else:
                        dis_strats += [sds]

                for t in cts:
                    dis_strats += [['scale',t,scale_factor]]

                n = Network(CONFIGS)
                n.set_distillation_strategies(dis_strats)
                n.calc_order('nb_neurons', droppable_scalable_layers)
                if np.sum(n.nb_neurons) < orig_net_size:
                    nets += [n]
        
            nets = sorted(nets)
            if CONFIGS['scale_limit_strat'] == 'distance':
                scale_networks_size_distance = CONFIGS['scale_network_size_distance']
                seed_nb_neurons = np.sum(ns.nb_neurons)
                lb = seed_nb_neurons*(1-scale_networks_size_distance)
                ub = seed_nb_neurons*(1+scale_networks_size_distance)
                nets = [n for n in nets if np.sum(n.nb_neurons) > lb and np.sum(n.nb_neurons) < ub]

            thread = threading.Thread(target=binary_search, args=(nets, 0, len(nets)-1, 2))
            thread.start()
            threads += [thread]
            nets_threads += [nets]
            #binary_search(nets, 0, len(nets)-1, phase=2)

        print(len(nets))
        for i, t in enumerate(threads):
            t.join()
            logging.info("Thread %d joined.", i)
        nets_phase = []
        for n in nets_threads:
            nets_phase += n
        all_nets += [nets_phase]
        logging.info('Phase {} finished.'.format(p+2))
        plot_data(nets, p, args.config.split('/')[1][:-5]+'.txt')


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
    
    nets = []

    for n in networks:
        if not os.path.exists(n.test_done_path):
            nets += [n]
            
    nets = nets[:8]
    print(len(nets))
    for n in nets:
        n.test()

        

    
def binary_search(networks, L, R, phase, d=0):
    if L>R:
        return
    elif L==R and networks[L].processed:
        return

    M = (L+R)//2
    net = networks[M]
    print(net.name)

    '''
    if net.name == 'dave.D.3.4.S.0.1.2.8':
        plot_data(networks)
        exit()
    '''
    if net.name in [ 'dave.D.3.4.S.1.9', 'dave.D.3.4.S.1.2.7.9.10']:
        plot_data(networks)
        exit()

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
        print('Net score', net.score_ra, net.accurate(2), net.score_veri, net.verifiable(2))
        if net.accurate(2) and not net.verifiable(2):
            binary_search(networks, L, M, d+1)
        elif not net.accurate(2) and net.verifiable(2):
            binary_search(networks, M, R, d+1)
        elif not net.accurate(2) and not net.verifiable(2):
            binary_search(networks, L, M, d+1)
            binary_search(networks, M, R, d+1)
    else:
        assert False


def plot_data(networks, phase, path):
    
    f = open(path,'a')
    f.write('temp_nd\n')
    for n in networks:
        if n.processed:
            for v in n.veri_res['neurify']:
                f.write("{}.iter.{},{}.neurify,{},{}\n".format(n.name,n.best_iter,v[0],v[1],v[2]))

    f.write('pref\n')
    for n in networks:
        if n.processed:
            f.write('{}.iter.{},0,0,{},0,0\n'.format(n.name,n.best_iter,n.score_ra))

    f.write('arch\n')
    for n in networks:
        if n.processed:
            f.write('{},,,{},,,,\n'.format(n.name, np.sum(n.nb_neurons)))



def main(args):
    configure(args)
    NAS_BS(args)


if __name__ == '__main__':
    main(_parse_args())

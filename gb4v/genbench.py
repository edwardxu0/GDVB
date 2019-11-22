import os
import numpy as np
import random
random.seed(0)

from itertools import combinations

from nn.onnxu import Onnxu
from gb4v.network import Network


def get_layers(onnx_net, configs):
    layers_with_ids = ['Conv', 'FC', 'Transpose', 'Flatten']
    layers = []
    for l in onnx_net.arch:
        if l.type in layers_with_ids:
            layers += [l]
    if configs['network_name'] == 'dave':
        layers = layers[1:]
    else:
        assert False
    return layers


def drop_scale_ids(layers):
    layers = layers[:-1]
    droppable_layers = ['Conv', 'FC']
    scalable_layers = ['Conv', 'FC']
    drop_ids = []
    scale_ids = []
    for i,l in enumerate(layers):
        if l.type in droppable_layers:
            drop_ids += [i]
        if l.type in scalable_layers:
            scale_ids += [i]
    return drop_ids, scale_ids


def calc_neurons(onnx_path, configs):
    onnx_net = Onnxu(onnx_path)
    droppable_scalable_layers = get_layers(onnx_net, configs)
    n = Network(configs)
    n.set_distillation_strategies([])
    n.calc_order('nb_neurons', droppable_scalable_layers)


def gen(args, configs):
    logger = configs['logger']
    
    source_net_onnx_path = Onnxu(configs['source_model'])
    source_net_layers = get_layers(source_net_onnx_path, configs)
    
    source_net = Network(configs)
    source_net.set_distillation_strategies([])
    source_net.calc_order('nb_neurons', source_net_layers)
    source_net_nb_neurons = np.sum(source_net.nb_neurons)
    
    drop_ids, scale_ids = drop_scale_ids(source_net_layers)

    nb_modi_layers = len(set(drop_ids).union(set(scale_ids)))

    logger.info('P1 drop layers.')


    nets = []
    # 1) drop layers
    trans = drop_ids
    trans_comb = []
    for i in range(len(trans)):
        trans_comb += combinations(trans, i+1)
    trans_comb = [list(x) for x in trans_comb]
    logger.info(f'# drop combs: {len(trans_comb)}')

    for ids in trans_comb:
        n = Network(configs)
        dis_strats = [['drop', x, 1] for x in ids]
        n.set_distillation_strategies(dis_strats)
        n.calc_order('nb_neurons', source_net_layers)
        nets += [n]
    logger.info(f'# NN: {len(nets)}')

    # 2) scale layers
    for _ in range(2):
        new_nets = []
        for dn in nets:
            s_ids = [x for x in scale_ids if x not in dn.drop_ids]
            trans = s_ids
            trans_comb = []
            for i in range(len(trans)):
                trans_comb += combinations(trans, i+1)
            trans_comb = [list(x) for x in trans_comb]

            logger.info(f'# scale combs: {len(trans_comb)}')

            for ids in trans_comb:
                n = Network(configs)
                dis_strats = [x for x in dn.dis_strats if x[0] == 'drop']
                
                for i in ids:
                    scale_factor = 0.5
                    for x in dn.scale_ids_factors:
                        if i == x[0]:
                            scale_factor *= x[1]
                            break
                    dis_strats += [['scale', i, scale_factor]]
                                         
                n.set_distillation_strategies(dis_strats)
                n.calc_order('nb_neurons', source_net_layers, True)
                #if np.sum(n.nb_neurons) < source_net_nb_neurons:
                new_nets += [n]
        nets += new_nets
    
    new_nets += [n for n in nets if np.sum(n.nb_neurons) < source_net_nb_neurons]
    nets = sorted(new_nets)

    logger.info(f'# NN: {len(nets)}')
    
    # factors

    parameters = {}

    '''
    # TODO: think about this
    max_nb_neurons = np.max([np.sum(x.nb_neurons) for x in nets])
    bins = [max_nb_neurons*(x+1)/neuron_bins for x in range(nb_neuron_bins)]
    '''
    
    nb_neuron_bins = configs['factor']['architecture']['neuron_bins']
    nb_neuron_bins_bases = [source_net_nb_neurons*(x+1)/nb_neuron_bins for x in range(nb_neuron_bins)]
    nb_neuron_bins = [[] for i in range(nb_neuron_bins)]
    
    layer_depth_step = configs['factor']['architecture']['layer_depth_step']
    nb_layer_bins_bases = [int((x+1)/layer_depth_step) for x in range(nb_modi_layers)]
    nb_layer_bins = [[] for i in range(len(nb_layer_bins_bases))]

    # TODO: fix this to generalize to all type of layers
    layer_type = configs['factor']['architecture']['layer_types']
    layer_type_bins = [[],[]]
    
    for n in nets:
        for i,x in enumerate(nb_neuron_bins_bases):
            if x > np.sum(n.nb_neurons):
                break
        nb_neuron_bins[i-1]+=[n]
        for i,x in enumerate(nb_layer_bins_bases):
            if x > len(n.nb_neurons):
                break
        nb_layer_bins[i-1]+=[n]

        isConv = False
        for l in n.layers:
            if l.type == 'Conv':
                isConv = True
        if isConv:
            layer_type_bins[0] += [n]
        else:
            layer_type_bins[1] += [n]

    print([len(x) for x in nb_neuron_bins])
    p = np.sum([1 if len(x)>0 else 0 for x in nb_neuron_bins])
    parameters['nb_neurons'] =p
    p = np.sum([1 if len(x)>0 else 0 for x in nb_layer_bins])
    parameters['nb_layers'] = p
    p = np.sum([1 if len(x)>0 else 0 for x in layer_type_bins])
    parameters['layer_types'] = p
    
    property_count = configs['factor']['property']['count']
    parameters['property_count'] = property_count
    
    epsilons = configs['factor']['property']['epsilons']
    parameters['epsilons_count'] = len(epsilons)
        
    weight_value_repeats = configs['factor']['other']['weight_value_repeats']
    parameters['weight_value_repeats'] = weight_value_repeats

    print(parameters)

    lines = [
        '[System]', 
        f'Name: {configs["network_name"]}',
        '',
        '[Parameter]']
    
    for x in parameters:
        xx = [str(xx) for xx in range(parameters[x])]
        y = ','.join(xx)
        lines += [f'{x}(int) : {y}']
    lines = [x +'\n' for x in lines]

    open('test.txt','w').writelines(lines)
    cmd = 'java -jar lib/acts_3.2.jar test.txt output1.txt'
    os.system(cmd)


def gen2(args, configs):
    logger = configs['logger']

    logger.info('Factors')
    # calculate parameters based on factors
    parameters = {'neurons':5,
               'fc_layers':4,
               'conv_layers':4,
               #'act_fun':3,
               'in_dim':4,
               'in_dom_size':4,
               'property':5,
               'epsilon':5}
    
    logger.info('Covering Array')

    # compute the covering array
    lines = [
        '[System]',
        f'Name: {configs["network_name"]}',
        '',
        '[Parameter]']
    
    for x in parameters:
        xx = [str(xx) for xx in range(parameters[x])]
        y = ','.join(xx)
        lines += [f'{x}(int) : {y}']
        
        #lines += [f'{x}(int) : {parameters[x]}']
    lines = [x +'\n' for x in lines]

    open('./tmp/ca_config.txt','w').writelines(lines)
    cmd = 'java  -Ddoi=2 -jar lib/acts_3.2.jar ./tmp/ca_config.txt ./tmp/ca.txt'
    os.system(cmd)
    
    vp_configs = []

    lines = open('./tmp/ca.txt','r').readlines()
    i = 0
    while i < len(lines):
        l = lines[i]
        if 'Number of configurations' in l:
            nb_tests = int(l.strip().split(' ')[-1])
            print(nb_tests)

        if 'Configuration #' in l:
            vp = []
            for j in range(2, len(parameters)):
                l = lines[j+i]
                vp += [int(l.strip().split('=')[-1])]
            vp_configs += [vp]
            i+=j
        i+=1
    assert len(vp_configs) == nb_tests

    '''
    vp_configs = []
    for x1 in range(4):
        for x2 in range(4):
            for x3 in range(4):
                for x4 in range(3):
                    for x5 in range(5):
                        for x6 in range(5):
                            for x7 in range(5):
                                    for x8 in range(5):
                                        vp_configs += [[x1,x2,x3,x4,x5,x6,x7,x8]]
    '''

    print(vp_configs)
    net_configs = []
    prop_configs = []
    print(net_configs)

    for vp in vp_configs:
        net_configs += [vp[:-2]]
        prop_configs += [vp[-2:]]

    net_configs = set(tuple(x) for x in net_configs)
    prop_configs = set(tuple(x) for x in prop_configs)

    
    # gen networks
    source_net_onnx_path = Onnxu(configs['source_model'])
    source_net_layers = get_layers(source_net_onnx_path, configs)
    
    source_net = Network(configs)
    source_net.set_distillation_strategies([])
    source_net.calc_order('nb_neurons', source_net_layers)
    source_net_nb_neurons = np.sum(source_net.nb_neurons)
    print(source_net_nb_neurons)
    
    #drop_ids, scale_ids = drop_scale_ids(source_net_layers)
    #nb_modi_layers = len(set(drop_ids).union(set(scale_ids)))

    fc_ids = [7,8,9,10]
    conv_ids = [0,1,2,3,4]
    fc_drop_factors = [0.75, 0.5, 0.25, 0]
    conv_drop_factors = [0.75, 0.5, 0.25, 0]
    scale_factors = [1,0.8,0.6,0.4,0.2]

    
    neurons = []
    nets = []
    for nc in net_configs:
        #print('net config:',nc)
        #print(nc[0])
        scale_factor = scale_factors[nc[0]]
        drop_fc_ids = random.sample(fc_ids, int(round(len(fc_ids)*fc_drop_factors[nc[1]])))
        #print(drop_fc_ids, fc_ids)
        drop_conv_ids = random.sample(conv_ids, int(round(len(conv_ids)*conv_drop_factors[nc[2]])))
        #print(drop_conv_ids, conv_ids)
        drop_ids = drop_fc_ids + drop_conv_ids
        scale_ids = set(fc_ids + conv_ids) - set(drop_ids)
        #print(drop_ids, scale_ids)

        n = Network(configs)
        dis_strats = [['drop', x, 1] for x in drop_ids]
        if scale_factor != 1:
            dis_strats += [['scale', x, 1] for x in scale_ids]
        #print(dis_strats)
        n.set_distillation_strategies(dis_strats)
        n.calc_order('nb_neurons', source_net_layers)


        if scale_factor != 1:
            neurons_drop_only = np.sum(n.nb_neurons)
            #print('neuron percent:', scale_factors)
            #print('neurons:', neurons_drop_only)

            scale_factor = (source_net_nb_neurons*scale_factor)/neurons_drop_only
            #print('factor:', scale_factor)
        
            n = Network(configs)
            dis_strats = [['drop', x, 1] for x in drop_ids]
            dis_strats += [['scale', x, scale_factor] for x in scale_ids]
            #print(dis_strats)
            n.set_distillation_strategies(dis_strats)
            n.calc_order('nb_neurons', source_net_layers)
            
        nets += [n]
        neurons += [np.sum(n.nb_neurons)]
        #print('neurons:', neurons)


    for n in nets:
        n.distill()
        exit()
        
        
    logger.info(f'# NN: {len(nets)}')

    print(neurons,np.mean(neurons))

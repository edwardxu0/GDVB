import os
import numpy as np
import random
import toml
import time
import pickle
import string
import datetime

from itertools import combinations

from .nn.onnxu import Onnxu
from .nn.layers import Dense, Conv
from .network import Network


def gen(configs):
    start_time = datetime.datetime.now()
    random.seed(configs['seed'])
    logger = configs['logger']

    if configs['name'] == 'acas':
        source_net_layers = acas_layers()
        input_shape = [1,5]
    else:
        source_net_onnx = Onnxu(configs['dnn']['onnx'])
        source_net_layers = get_layers(configs, source_net_onnx)
        input_shape = source_net_onnx.input_shape

    source_net = Network(configs, None)
    source_net.set_distillation_strategies([])

    source_net.calc_order('nb_neurons', source_net_layers, input_shape)
    source_net_neurons_per_layer = source_net.nb_neurons
    source_net_nb_neurons = np.sum(source_net_neurons_per_layer)

    logger.info('Computing Factors')
    # calculate parameters based on factors
    net_name = configs['name']

    # calculate fc conv ids
    conv_ids = []
    fc_ids = []
    for i,x in enumerate(source_net_layers):
        if isinstance(x, Conv):
            conv_ids+=[i]
        elif isinstance(x, Dense):
            fc_ids += [i]
    fc_ids = fc_ids[:-1]

    # load parameters
    parameters = {}

    for key in configs['ca']['parameters']['level']:
        level_size = configs['ca']['parameters']['level'][key]
        if level_size == 1:
            level_min = configs['ca']['parameters']['range'][key][0]
            level_max = configs['ca']['parameters']['range'][key][1]
            assert level_min == level_max
            level = np.array([level_min])
            pass
        else:
            if key in ['prop']:
                level = np.arange(0,level_size,1)
            else:
                level_min = configs['ca']['parameters']['range'][key][0]
                level_max = configs['ca']['parameters']['range'][key][1]
                level_range = level_max - level_min
                level_step = level_range/(level_size-1)
                assert level_range > 0
                level = np.array([x*level_step+level_min for x in range(level_size)])
        parameters[key] = level

        assert len(parameters[key]) == level_size
    #print(parameters)

    logger.info('Covering Array')

    # compute the covering array
    lines = [
        '[System]',
        f'Name: {configs["name"]}',
        '',
        '[Parameter]']
    
    for key in parameters:
        bits = [str(x) for x in range(len(parameters[key]))]
        bits = ','.join(bits)
        lines += [f'{key}(int) : {bits}']

    if 'constraints' in configs['ca']:
        lines += ['[Constraint]']
        constraints = configs['ca']['constraints']['value']
        for con in constraints:
            lines += [con]

    lines = [x +'\n' for x in lines]

    strength = configs['ca']['strength']

    open('./tmp/ca_config.txt','w').writelines(lines)
    cmd = f'java  -Ddoi={strength} -jar lib/acts_3.2.jar ./tmp/ca_config.txt ./tmp/ca.txt > /dev/null'
    os.system(cmd)
    os.remove("./tmp/ca_config.txt")

    vp_configs = []
    lines = open('./tmp/ca.txt','r').readlines()
    os.remove("./tmp/ca.txt")

    i = 0
    while i < len(lines):
        l = lines[i]
        if 'Number of configurations' in l:
            nb_tests = int(l.strip().split(' ')[-1])
            logger.info(f'# Tests: {nb_tests}')

        if 'Configuration #' in l:
            vp = []
            for j in range(len(parameters)):
                l = lines[j+i+2]
                vp += [int(l.strip().split('=')[-1])]
            assert len(vp) == len(parameters)
            vp_configs += [vp]
            i+=j+2
        i+=1
    assert len(vp_configs) == nb_tests
    vp_configs = sorted(vp_configs)

    vp_configs_ = []
    for vpc in vp_configs:
        assert len(vpc) == len(parameters)
        tmp = {}
        for i,key in enumerate(parameters):
            tmp[key] = vpc[i]
        vp_configs_ += [tmp]
    vp_configs = vp_configs_

    '''
    for vpc in vp_configs:
        print(vpc)
    '''

    nets = []
    for vpc in vp_configs:
        # generate network
        neuron_scale_factor = parameters['neu'][vpc['neu']]
        drop_fc_ids = sorted(random.sample(fc_ids, int(round(len(fc_ids)*(1-parameters['fc'][vpc['fc']])))))
        if 'conv' in parameters:
            drop_conv_ids = sorted(random.sample(conv_ids, int(round(len(conv_ids)*(1-parameters['conv'][vpc['conv']])))))
            drop_ids = drop_fc_ids + drop_conv_ids
        else:
            drop_ids = drop_fc_ids
        scale_ids = set(fc_ids + conv_ids) - set(drop_ids)

        n = Network(configs, vpc)
        dis_strats = [['drop', x] for x in drop_ids]

        # calculate data transformaions, input demensions
        if not configs['name'] == 'acas':
            transform = n.distillation_config['distillation']['data']['transform']['student']
            height = transform['height']
            width = transform['width']
            assert height == width
            id_f = parameters['idm'][vpc['idm']]

            new_height = int(round(np.sqrt(height*width*id_f)))
            transform['height'] = new_height
            transform['width'] = new_height
            if new_height != height:
                dis_strats += [['scale_input', new_height/height]]

            mean = transform['mean']
            max_value = transform['max_value']
            ids_f = parameters['ids'][vpc['ids']]

            transform['mean'] = [float(x*ids_f) for x in mean]
            transform['max_value'] = float(max_value*ids_f)

            if source_net_onnx.input_format == 'NCHW':
                nb_channel = source_net_onnx.input_shape[0]
            elif source_net_onnx.input_format == 'NHWC':
                nb_channel = source_net_onnx.input_shape[2]
            else:
                assert False
        else:
            ids_f = parameters['ids'][vpc['ids']]
            n.distillation_config['distillation']['data']['train']['student']['path'] = f'data/acas/acas_train_{ids_f}.npy'
            n.distillation_config['distillation']['data']['validation']['student']['path'] = f'data/acas/acas_valid_{ids_f}.npy'
            #print(ids_f)
            #print(n.distillation_config)

        #print('before:')
        #print(dis_strats)
        n.set_distillation_strategies(dis_strats)
        if not configs['name'] == 'acas':
            input_shape = [nb_channel, new_height, new_height]
        n.calc_order('nb_neurons', source_net_layers, input_shape)

        neurons_drop_only = np.sum(n.nb_neurons)
        #print('neuron percent:', scale_factors)
        #print('neurons:', neurons_drop_only)

        neuron_scale_factor = (source_net_nb_neurons*neuron_scale_factor)/neurons_drop_only
        #assert neuron_scale_factor != 1
        if neuron_scale_factor != 1:
            #print('factor:', scale_factor)
            for x in scale_ids:
                if round(source_net_neurons_per_layer[x] * neuron_scale_factor) == 0 :
                    neuron_scale_factor2 = 1/source_net_neurons_per_layer[x]
                    dis_strats += [['scale', x, neuron_scale_factor2]]
                else:
                    dis_strats += [['scale', x, neuron_scale_factor]]
            #print('after:')
            #print(dis_strats)

            n = Network(configs, vpc)

            if not configs['name'] == 'acas':
                n.distillation_config['distillation']['data']['transform']['student'] = transform
            n.set_distillation_strategies(dis_strats)
            n.calc_order('nb_neurons', source_net_layers, input_shape)
        nets += [n]

    logger.info(f'# NN: {len(nets)}')

    if configs['task'] == 'gen_ca':
        pass

    elif configs['task'] == 'train':
        logger.info('Training ...')
        train(nets, logger)

    elif configs['task'] == 'gen_props':
        logger.info('Generating properties ...')
        gen_props(nets, parameters, configs, logger)

    elif configs['task'] == 'verify':
        logger.info('Verifying ...')
        verify(nets, configs, parameters, logger)

    elif configs['task'] == 'analyze':
        analyze(nets, configs)

    else:
        raise Exception("Unknown task.")

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f'Spent {duration} seconds.')


def get_layers(configs, onnx_dnn):
    supported_layers = configs['dnn']['supported_layers']
    start_layer = configs['dnn']['start_layer']
    layers = []
    for l in onnx_dnn.arch:
        if l.type in supported_layers:
            layers += [l]
    return layers[start_layer:]


def acas_layers():
    layers = [
        Dense(50, np.zeros((50,5)), np.zeros((50)), 5),
        Dense(50, np.zeros((50,50)), np.zeros((50)), 50),
        Dense(50, np.zeros((50,50)), np.zeros((50)), 50),
        Dense(50, np.zeros((50,50)), np.zeros((50)), 50),
        Dense(50, np.zeros((50,50)), np.zeros((50)), 50),
        Dense(50, np.zeros((50,50)), np.zeros((50)), 50),
        Dense(5, np.zeros((5,50)), np.zeros((5)), 50)
    ]
    return layers


def train(nets, logger):
    for n in nets:
        logger.info(f'Training network {n.name} ...')
        n.train()
        logger.info(f'Training network {n.name} done.')
        #break


def gen_props(nets, parameters, configs, logger):
    logger.info('Generating properties ...')
    for n in nets:
        data_config = open(configs['dnn']['data_config'],'r').read()
        data_config = toml.loads(data_config)
        transform = n.distillation_config['distillation']['data']['transform']['student']
        data_config['transform'] = transform

        formatted_data = toml.dumps(data_config)
        with open('./tmp/data.toml', 'w') as f:
            f.write(formatted_data)

        prop_dir = f'{configs["props_dir"]}/{n.name}/'

        eps = (parameters['eps']*configs['verify']['eps'])[n.vpc['eps']]

        if 'dave' in configs['name']:
            cmd = f'python ./tools/generate_dave_properties.py ./tmp/data.toml {prop_dir} -g 15'
        elif 'mcb' in configs['name'] or 'ms' in configs['name']:
            cmd = f'python ./tools/generate_mnist_properties.py ./tmp/data.toml {prop_dir}'
        else:
            raise NotImplementedError
        cmd += f' -e {eps} -N {len(parameters["prop"])}'
        os.system(cmd)
        os.system('rm ./tmp/data.toml')


def verify(nets, configs, parameters, logger):
    for n in nets:
        logger.info(f'Verifying network {n.name} ...')
        n.verify(parameters,logger)
        logger.info(f'Verifying network {n.name} done.')
        #break


def analyze(nets, configs):
    verifiers = configs['verify']['verifiers']
    results = {x:{} for x in verifiers}

    for n in nets:
        for v in verifiers:
            vpc = ''.join([str(n.vpc[x]) for x in n.vpc])
            log = f"{n.config['veri_log_dir']}/{vpc}_{v}.out"
            if not os.path.exists(log):
                res = 'torun'
                print(res, log)
            else:
                rlines = [x for x in reversed(open(log,'r').readlines())]

                res = None
                v_time = None

                for i,l in enumerate(rlines):
                    if l.startswith('INFO'):# or l.startswith('DEBUG'):
                        continue 

                    if '[STDERR]:Error: GLP returned error 5 (GLP_EFAIL)' in l:
                        res = 'error'
                        print(log)
                        break

                    if "*** Error in `python':" in l:
                        res = 'error'
                        print(log)
                        break

                    if 'Cannot serialize protocol buffer of type ' in l:
                        res = 'error'
                        print(log)
                        break

                    if 'Timeout' in l:
                        res = 'timeout'
                        break

                    elif 'Out of Memory' in l:
                        res = 'memout'
                        break

                    if l.startswith('  result: '):
                        if 'Unsupported' in l or 'not support' in l or 'Unknown MIPVerify' in l or 'Unknown property check result' in l:
                            res = 'unsup'
                            break
                        elif 'NeurifyError' in l or 'PlanetError' in l:
                            res = 'error'
                            break
                        else:
                            res = l.strip().split(' ')[-1]
                            v_time = float(rlines[i-1].strip().split(' ')[-1])
                            break

                # remove this
                if i == len(rlines)-1:
                    res = 'running'
                    print(res, log)

            if res not in ['sat','unsat']:
                v_time = 14400

            assert res in ['sat','unsat','unknown','timeout','memout','error', 'unsup', 'running', 'torun'], log
            results[v][vpc] = [res,v_time]

    assert len(set([len(results[x]) for x in results.keys()])) == 1

    with open(f'./results/verification_results_{configs["name"]}_{configs["seed"]}.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    config_dist = {}
    for n in nets:
        config_dist[''.join([str(n.vpc[x]) for x in n.vpc])] = n.vpc

    with open(f'./results/config_dist_{configs["name"]}_{configs["seed"]}.pickle', 'wb') as handle:
        pickle.dump(config_dist, handle, protocol=pickle.HIGHEST_PROTOCOL)


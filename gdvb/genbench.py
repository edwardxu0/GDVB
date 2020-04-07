import os
import numpy as np
import random
import toml
import time
import pickle
import string
import datetime

from itertools import combinations

from gdvb.nn.onnxu import Onnxu
from gdvb.nn.layers import Dense, Conv
from gdvb.network import Network


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

    logger.info('Factors')
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

            '''
            containsScale = False
            for i in dis_strats:
                if i[0] == 'scale':
                    containsScale = True
            print(containsScale)
            '''
        else:
            '''
            containsScale = False
            for i in dis_strats:
                if i[0] == 'scale':
                    containsScale = True
            print(containsScale)
            if not containsScale:
                print(dis_strats)
            '''
            pass
            
        '''
        if n.input_too_small():
            ca_lines_constraints += [f'conv_layers={vpc["conv_layers"]} => in_dim!={vpc["in_dim"]}']
            #refine_constraints = True

        if n.untrainable(configs['parameter_limit']):
            if conv_drop_factors is None and fc_drop_factors is None:
                assert False
            elif conv_drop_factors is None:
                ca_lines_constraints += [f'fc_layers={vpc["fc_layers"]} => in_dim!={vpc["in_dim"]}']
            elif fc_drop_factors is None:
                ca_lines_constraints += [f'conv_layers={vpc["conv_layers"]} => in_dim!={vpc["in_dim"]}']
            else:
                ca_lines_constraints += [f'(fc_layers={vpc["fc_layers"]} && conv_layers={vpc["conv_layers"]}) => in_dim!={vpc["in_dim"]}']
            refine_constraints = True
        '''
        nets += [n]

    logger.info(f'# NN: {len(nets)}')

    # print pplot
    '''
    for n in nets:
        a = n.vpc
        print(f'[{np.sum(n.nb_neurons)},{a["fc"]},{a["conv"]},{parameters["idm"][a["idm"]]*784:.0f},{parameters["ids"][a["ids"]]:.1f},{parameters["eps"][a["eps"]]:.1f},{a["prop"]}],')
    exit()
    '''
    gen_props(nets, parameters, configs, logger)
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(duration)
    
    #TODO: dont distill repeated networks
    if configs['task'] == 'train':
        logger.info('Training ...')
        for n in nets:
            print(n.name)
            n.distill()

    elif configs['task'] == 'gen_props':
        gen_props(nets, configs)

    elif configs['task'] == 'verify':
        logger.info('Verifying ...')
        verifiers = configs['verify']['verifiers']
        time_limit = configs['verify']['time']
        memory_limit = configs['verify']['memory']

        lines = []
        count = 0
        for v in verifiers:
            if 'eran' in v:
                v_name = 'eran'
                verifier_parameters = f'--eran.domain {v.split("_")[1]}'
            elif v == 'bab_sb':
                v_name = 'bab'
                verifier_parameters = '--bab.smart_branching'
            else:
                v_name = v
                verifier_parameters = ""

            for n in nets:
                if not configs['name'] == 'acas':
                    prop_dir = f'{configs["props_dir"]}/{n.name}/'
                else:
                    prop_dir = 'props/acas/'
                #prop_dir = f'{configs["props_dir"]}/{configs["network_name"]}.{input_dimension_levels[n.vpc["in_dim"]]}.{input_domain_size_levels[n.vpc["in_dom_size"]]}.{epsilon_levels[n.vpc["epsilon"]]}'
                eps = (parameters['eps']*configs['verify']['eps'])[n.vpc['eps']]
                prop_levels = sorted([x for x in os.listdir(prop_dir) if '.py' in x])
                prop_levels = [ x for x in prop_levels if str(eps) in '.'.join(x.split('.')[2:-1])]
                prop = prop_levels[n.vpc['prop']]
                
                cmd = f'python ../dnnv/tools/resmonitor.py -T {time_limit} -M {memory_limit}'
                cmd += f' python -m dnnv {n.dis_model_path} {prop_dir}/{prop} --{v_name} {verifier_parameters} --debug'

                #NODES = ['slurm1', 'slurm2', 'slurm3', 'slurm4', 'slurm5']
                NODES = configs['dispatch']['slurm']['nodes']
                #TASK_NODE = {'slurm1':7,'slurm2':7,'slurm3':7,'slurm4':7,'slurm5':3}
                TASK_NODE = configs['dispatch']['slurm']['task_per_node']
                reservation = configs['dispatch']['slurm']['reservation']
                #print(NODES,TASK_NODE,reservation)

                #NODES = [f'cortado0{x}' for x in range(1,8)]
                #TASK_NODE = {x:7 for x in NODES}

                #print(TASK_NODE)
                count += 1
                logger.info(f'Verifying ...{count}/{len(verifiers)*len(nets)}')
                vpc = ''.join([str(n.vpc[x]) for x in n.vpc])

                veri_log = f'{n.config["veri_log_dir"]}/{vpc}_{v}.out'
                if os.path.exists(veri_log):
                    lines = open(veri_log, 'r').readlines()

                    #TODO edit this
                    print('done')
                    continue
                
                    rerun = False
                    for l in lines:
                        if 'Traceback' in l:
                            rerun = True                            
                            break
                    '''
                    if error:
                        for l in lines:
                            if 'FileNotFoundError' in l:
                                rerun = False
                                break
                            if 'Unsupported layer type' in l and 'reluplex' in log:
                                rerun = False
                                break
                    '''

                    if not rerun:
                        print('done')
                        continue
                    else:
                        print('rerun')

                while(True):
                    node_avl_flag = False
                    tmp_file = './tmp/'+''.join(random.choice(string.ascii_lowercase) for i in range(16))
                    node_ran = ''.join([x for x in NODES[0] if not x.isdigit()])
                    #sqcmd = f'squeue | grep {node_ran} > {tmp_file}'
                    sqcmd = f'squeue -u dx3yy > {tmp_file}'
                    time.sleep(2)
                    os.system(sqcmd)
                    sq_lines = open(tmp_file, 'r').readlines()[1:]
                    os.remove(tmp_file)
                    
                    nodes_avl = {}
                    for node in NODES:
                        nodes_avl[node] = 0

                    nodenodavil_flag = False
                    for l in sq_lines:
                        if 'ReqNodeNotAvail' in l and 'dx3yy' in l or ('Priority' in l and 'GB_D' not in l) or ('None' in l and 'GB_D' not in l):
                            nodenodavil_flag = True
                            break
                    
                        if ' R ' in l and l != '':
                            node = l.strip().split(' ')[-1]
                            if node in NODES:
                                nodes_avl[node] += 1

                    if nodenodavil_flag == True:
                        print('node unavialiable. waiting ...')
                        continue

                    #print(nodes_avl)
            
                    for na in nodes_avl:
                        #if nodes_avl[na] < TASK_NODE[na]:
                        if nodes_avl[na] < 7:
                            node_avl_flag = True
                            break
                    if node_avl_flag:
                        break

                tmp_dir = f'./tmp/{configs["name"]}_{configs["seed"]}_{vpc}_{v}'
                lines = ['#!/bin/sh',
                         f'#SBATCH --job-name=GB_{v}',
                         f'#SBATCH --mem={memory_limit}',
                         f'#SBATCH --output={n.config["veri_log_dir"]}/{vpc}_{v}.out',
                         f'#SBATCH --error={n.config["veri_log_dir"]}/{vpc}_{v}.out',
                         f'export GRB_LICENSE_FILE="/p/d4v/dx3yy/Apps/gurobi_keys/`hostname`.gurobi.lic"',
                         f'export TMPDIR={tmp_dir}',
                         f'mkdir $TMPDIR',
                         f'echo $TMPDIR',
                         cmd,
                         f'rm -rf $TMPDIR'
                ]
                lines = [x+'\n' for x in lines]
                slurm_path = os.path.join(n.config['veri_slurm_dir'],f'{vpc}_{v}.slurm')
                open(slurm_path,'w').writelines(lines)

                task = f'sbatch -w {na} --reservation={reservation} {slurm_path}'
                print(task)
                os.system(task)
                

    elif configs['task'] == 'ana_res':
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

        with open(f'results/data/verification_results_{configs["name"]}_{configs["seed"]}.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        config_dist = {}
        for n in nets:
            config_dist[''.join([str(n.vpc[x]) for x in n.vpc])] = n.vpc

        with open(f'results/data/config_dist_{configs["name"]}_{configs["seed"]}.pickle', 'wb') as handle:
            pickle.dump(config_dist, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        assert False


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
            cmd = f'python ../r4v/tools/generate_dave_properties.py ./tmp/data.toml {prop_dir} -g 15'
        elif 'mcb' in configs['name']:
            cmd = f'python ../r4v/tools/generate_mnist_properties.py ./tmp/data.toml {prop_dir}'
        cmd += f' -e {eps} -N {len(parameters["prop"])}'
        os.system(cmd)
        os.system('rm ./tmp/data.toml')

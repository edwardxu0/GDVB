import os
import numpy as np
import random
import toml
import time
random.seed(0)

from itertools import combinations

import nn
from nn.onnxu import Onnxu
from gb4v.network import Network


def gen(args, configs):
    logger = configs['logger']

    source_net_onnx = Onnxu(configs['dnn']['onnx'])
    source_net_layers = get_layers(configs, source_net_onnx)
    
    source_net = Network(configs, None)
    source_net.set_distillation_strategies([])

    source_net.calc_order('nb_neurons', source_net_layers, source_net_onnx.input_shape)
    source_net_neurons_per_layer = source_net.nb_neurons
    source_net_nb_neurons = np.sum(source_net_neurons_per_layer)

    logger.info('Factors')
    # calculate parameters based on factors
    net_name = configs['name']

    # calculate fc conv ids
    conv_ids = []
    fc_ids = []
    for i,x in enumerate(source_net_layers):
        if isinstance(x, nn.layers.Conv):
            conv_ids+=[i]
        elif isinstance(x, nn.layers.Dense):
            fc_ids += [i]
    fc_ids = fc_ids[:-1]

    # load parameters
    parameters = {}
    
    for key in configs['ca']['parameters']:
        level_size = configs['ca']['parameters'][key]

        if key in ['fc','conv']:
            level = np.arange(0,1+1/(level_size-1),1/(level_size-1))
        elif key in ['prop']:
            level = np.arange(0,level_size,1)
        else:
            level = np.arange(1/level_size,1+1/level_size,1/level_size)
        parameters[key] = level
        assert len(parameters[key]) == level_size
    
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
        dis_strats = [['drop', x, 1] for x in drop_ids]
        if neuron_scale_factor != 1:
            dis_strats += [['scale', x, 1] for x in scale_ids]


        # calculate data transformaions, input demensions
        transform = n.distillation_config['distillation']['data']['transform']['student']
        height = transform['height']
        width = transform['width']
        assert height == width
        id_f = parameters['idm'][vpc['idm']]

        new_height = int(round(np.sqrt(height*width*id_f)))
        transform['height'] = new_height
        transform['width'] = new_height
        if new_height != height:
            dis_strats += [['scale_input', new_height/height, None]]

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
        input_shape = [nb_channel, new_height, new_height]

        n.set_distillation_strategies(dis_strats)                                   
        n.calc_order('nb_neurons', source_net_layers, input_shape)

        neurons_drop_only = np.sum(n.nb_neurons)
        #print('neuron percent:', scale_factors)
        #print('neurons:', neurons_drop_only)

        neuron_scale_factor = (source_net_nb_neurons*neuron_scale_factor)/neurons_drop_only

        if neuron_scale_factor != 1:
            #print('factor:', scale_factor)
            n = Network(configs, vpc)
            dis_strats = [['drop', x, 1] for x in drop_ids]
            # TODO: roundding up if 0 neurons or kernels in r4v ; done
            for x in scale_ids:
                if round(source_net_neurons_per_layer[x] * neuron_scale_factor) == 0 :
                    neuron_scale_factor2 = 1/source_net_neurons_per_layer[x]
                    dis_strats += [['scale', x, neuron_scale_factor2]]
                else:
                    dis_strats += [['scale', x, neuron_scale_factor]]
            #print(dis_strats)
            n.name+='_'+str(neuron_scale_factor)[:8]
            n.set_distillation_strategies(dis_strats)
            input_shape = [nb_channel, new_height, new_height]
            n.calc_order('nb_neurons', source_net_layers, input_shape)

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

    ms2= []

    #TODO: dont distill repeated networks
    if args.task == 'train':
        logger.info('Training ...')
        for n in nets:
            print(n.name)
            if n.name in ms2:
                print('repeat')
            ms2 +=[n.name]
            n.distill()

    elif args.task == 'gen_prop':
        logger.info('Generating properties ...')
        for n in nets:
            data_config = open(configs['data_config'],'r').read()
            data_config = toml.loads(data_config)
            transform = n.distillation_config['distillation']['data']['transform']['student']
            data_config['transform'] = transform
            formatted_data = toml.dumps(data_config)
            with open('tmp/data.toml', 'w') as f:
                f.write(formatted_data)

            prop_dir = f'{configs["props_dir"]}/{n.name}/'
            #prop_dir = f'{configs["props_dir"]}/{configs["network_name"]}.{input_dimension_levels[n.vpc["in_dim"]]}.{input_domain_size_levels[n.vpc["in_dom_size"]]}.{epsilon_levels[n.vpc["epsilon"]]}'

            if configs['network_name'] == 'dave':
                cmd = f'python ../r4v/tools/generate_dave_properties.py tmp/data.toml {prop_dir} -g 15'
            elif configs['network_name'] in ['mnist_conv_super', 'mnist_dense50x2','mnist_conv_big']:
                cmd = f'python ../r4v/tools/generate_mnist_properties.py tmp/data.toml {prop_dir}'
            cmd += f' -e {epsilon_levels[n.vpc["epsilon"]]} -N {len(property_levels)}'
            os.system(cmd)

    elif args.task == 'verify':
        logger.info('Verifying ...')
        verifiers = ['eran','planet','reluplex','neurify','mipverify']
        lines = []
        count = 0
        for v in verifiers:
            for n in nets:
                prop_dir = f'{configs["props_dir"]}/{n.name}/'
                #prop_dir = f'{configs["props_dir"]}/{configs["network_name"]}.{input_dimension_levels[n.vpc["in_dim"]]}.{input_domain_size_levels[n.vpc["in_dom_size"]]}.{epsilon_levels[n.vpc["epsilon"]]}'
                prop_levels = sorted([x for x in os.listdir(prop_dir) if '.py' in x])
                prop = prop_levels[n.vpc['property']]
                cmd = f'python ../dnna/tools/resmonitor.py -T 14400 -M 64G python -m dnnv {n.dis_model_path} {prop_dir}/{prop} --{v}'
                
                NODES = ['slurm1', 'slurm2', 'slurm3', 'slurm4', 'slurm5']
                TASK_NODE = {'slurm1':7,'slurm2':7,'slurm3':7,'slurm4':7,'slurm5':3}
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
                    task = 'squeue -u dx3yy > tmp/squeue_results.txt'
                    time.sleep(5)
                    os.system(task)
                    sq_lines = open('tmp/squeue_results.txt', 'r').readlines()[1:]
                    nodes_avl = {}
                    for node in NODES:
                        nodes_avl[node] = 0

                    nodenodavil_flag = False
                    for l in sq_lines:
                        if 'ReqNodeNotAvail' in l and 'dx3yy' in l:
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
                        if nodes_avl[na] < TASK_NODE[na]:
                            node_avl_flag = True
                            break
                    if node_avl_flag:
                        break

                tmp_dir = f'./tmp/{configs["network_name"]}_{vpc}_{v}'
                lines = ['#!/bin/sh',
                         '#SBATCH --job-name=GB_V',
                         f'#SBATCH --output={n.config["veri_log_dir"]}/{vpc}_{v}.out',
                         f'#SBATCH --error={n.config["veri_log_dir"]}/{vpc}_{v}.out',
                         f'export TMPDIR={tmp_dir}',
                         f'mkdir $TMPDIR',
                         f'echo $TMPDIR',
                         cmd,
                         f'rm -rf $TMPDIR'
                ]
                lines = [x+'\n' for x in lines]
                slurm_path = os.path.join(n.config['veri_slurm_dir'],'{}_{}.slurm'.format(vpc, v),)
                open(slurm_path,'w').writelines(lines)
                
                task = 'sbatch -w {} --reservation=dls2fc_7 {}'.format(na, slurm_path)
                os.system(task)
                

    elif args.task == 'ana_res':
        verifiers = ['eran','planet','reluplex','neurify','mipverify']
        results = {x:{} for x in verifiers}

        for n in nets:
            for v in verifiers:
                vpc = ''.join([str(n.vpc[x]) for x in n.vpc])
                #log = f"{n.config['veri_log_dir']}/{vpc}_{v}.out"
                log = f"res.mnist_conv_big/veri_log0/{vpc}_{v}.out"
                lines = open(log,'r').readlines()

                error = False
                for l in lines:
                    if 'Traceback' in l:
                        error = True
                        break

                if error:
                    unknown_error = True
                    for l in lines:
                        if 'FileNotFoundError' in l:
                            unknown_error = False
                            break
                        if 'Unsupported layer type' in l and 'reluplex' in log:
                            unknown_error = False
                            break

                    if unknown_error:
                        print('Error:', log)
                    results[v][vpc] = ['error', 14400]
                    
                else:
                    if 'Timeout' in lines[-1]:
                        #print('Timeout', log)
                        results[v][vpc] = ['timeout', 14400]
                    elif 'Out of Memory' in lines[-1]:
                        #print('Memout', log)
                        results[v][vpc] = ['memout', 14400]
                    else:
                        lines = lines[-10:]
                        for i,l in enumerate(lines):
                            if 'result'in l:
                                if 'Unsupported' in l or 'not supported' in l or 'Unknown MIPVerify result' in l or 'Unknown property check result' in l :
                                    res = 'unsup'
                                elif 'OSError' in l:
                                    res = 'error'
                                else:
                                    res = lines[i].strip().split(' ')[-1]
                                v_time = float(lines[i+1].strip().split(' ')[-1])
                                results[v][vpc] = [res,v_time]
                                break
                if res not in ['sat','unsat','unknown','timeout','memout','error', 'unsup']:
                    print(res,log)
                    exit()
        '''
        untrained_tests = ['91003', '81003', '71003', '61003', '51002', '34030', '23001', '12031']
        print(untrained_tests)
        for v_k in results:
            assert len(results[v_k]) == len(nets)
            for vpc_k in results[v_k]:
                if vpc_k[:-2] in untrained_tests:
                    results[v_k][vpc_k] = ['untrain','']
                    #print(vpc_k, results[v_k][vpc_k])
        '''
                    
        print(results)


        print('| Verifier | SCR | PAR-2 |')
        for v_k in results:
            sums = []
            scr = 0
            for vpc_k in results[v_k]:
                res = results[v_k][vpc_k][0]
                vtime = results[v_k][vpc_k][1]

                if res in ['sat','unsat']:
                    scr += 1
                    sums += [vtime]
                elif res in ['unknown','timeout','memout','error','unsup']:
                    sums += [14400*2]
                elif res in ['untrain']:
                    pass
                else:
                    assert False, res
                

            par_2 = int(round(np.mean(np.array(sums))))
            
            print(f'|{v_k}|{scr}|{par_2}|')

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

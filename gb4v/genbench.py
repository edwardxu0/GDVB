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


def get_layers(onnx_net, configs):
    layers_with_ids = ['Conv', 'FC', 'Transpose', 'Flatten']
    layers = []
    for l in onnx_net.arch:
        if l.type in layers_with_ids:
            layers += [l]
    if configs['network_name'] == 'dave':
        layers = layers[1:]
    elif configs['network_name'] == 'mnist_conv_super':
        layers = layers[1:]
    elif configs['network_name'] == 'mnist_dense50x2':
        pass
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
    onnx_net = Onnxu(onnx_path, configs['image_shape_mode'])
    droppable_scalable_layers = get_layers(onnx_net, configs)
    n = Network(configs)
    n.set_distillation_strategies([])
    n.calc_order('nb_neurons', droppable_scalable_layers)


def gen(args, configs):
    logger = configs['logger']

    source_net_onnx_path = Onnxu(configs['source_model'], configs['image_shape_mode'])
    source_net_layers = get_layers(source_net_onnx_path, configs)
    
    source_net = Network(configs, None)
    source_net.set_distillation_strategies([])
    #print(source_net_layers[0].in_shape)
    source_net.calc_order('nb_neurons', source_net_layers, configs['image_shape'][1:])
    source_net_neurons_per_layer = source_net.nb_neurons
    source_net_nb_neurons = np.sum(source_net_neurons_per_layer)
    #print(source_net_nb_neurons)
    
    #drop_ids, scale_ids = drop_scale_ids(source_net_layers)
    #nb_modi_layers = len(set(drop_ids).union(set(scale_ids)))
    
    logger.info('Factors')
    # calculate parameters based on factors
    net_name = configs['network_name']
    conv_ids = []
    fc_ids = []
    for i,x in enumerate(source_net_layers):
        if isinstance(x, nn.layers.Conv):
            conv_ids+=[i]
        elif isinstance(x, nn.layers.Dense):
            fc_ids += [i]
    fc_ids = fc_ids[:-1]

    if net_name == 'dave':
        assert conv_ids == [0,1,2,3,4]
        assert fc_ids == [7,8,9,10]
        
        parameters = {'neurons':10,
                   'fc_layers':5,
                   'conv_layers':6,
                   #'act_fun':3,
                   'in_dim':4,
                   'in_dom_size':4,
                   'property':5,
                   'epsilon':5}

        neuron_scale_factors = np.arange(0.1,1.1,0.1)
        assert len(neuron_scale_factors) == parameters['neurons']
        fc_drop_factors = np.flip(np.arange(0,1.25,0.25))
        assert len(fc_drop_factors) == parameters['fc_layers']
        conv_drop_factors = np.flip(np.arange(0,1.2,0.2))
        assert len(conv_drop_factors) == parameters['conv_layers']
        input_dimension_levels = np.arange(0.25,1.25,0.25)
        assert len(input_dimension_levels) == parameters['in_dim']
        input_domain_size_levels = np.arange(0.25,1.25,0.25)
        assert len(input_domain_size_levels) == parameters['in_dom_size']

        property_levels = np.arange(0,5,1)
        assert len(property_levels) == parameters['property']
        epsilon_levels = np.arange(0.4,2.4,0.4)
        assert len(epsilon_levels) == parameters['epsilon']

    elif net_name == 'mnist_conv_super':
        assert conv_ids == [0,1,2,3]
        assert fc_ids == [6,7]
        
        parameters = {'neurons':10,
                   'fc_layers':3,
                   'conv_layers':5,
                   #'act_fun':3,
                   'in_dim':4,
                   'in_dom_size':4,
                   'property':5,
                   'epsilon':5}

        neuron_scale_factors = np.arange(0.1,1.1,0.1)
        assert len(neuron_scale_factors) == parameters['neurons']
        fc_drop_factors = np.flip(np.arange(0,1.5,0.5))
        assert len(fc_drop_factors) == parameters['fc_layers']
        conv_drop_factors = np.flip(np.arange(0,1.25,0.25))
        assert len(conv_drop_factors) == parameters['conv_layers']
        input_dimension_levels = np.arange(0.25,1.25,0.25)
        assert len(input_dimension_levels) == parameters['in_dim']
        input_domain_size_levels = np.arange(0.25,1.25,0.25)
        assert len(input_domain_size_levels) == parameters['in_dom_size']

        property_levels = np.arange(0,5,1)
        assert len(property_levels) == parameters['property']
        epsilon_levels = np.arange(0.4,2.4,0.4)
        assert len(epsilon_levels) == parameters['epsilon']

    elif net_name == 'mnist_dense50x2':
        assert conv_ids == []
        assert fc_ids == [1,2]
        
        parameters = {'neurons':4,
                   'fc_layers':3,
                   #'conv_layers':0,
                   #'act_fun':3,
                   'in_dim':4,
                   'in_dom_size':4,
                   'property':5,
                   'epsilon':5}

        neuron_scale_factors = np.arange(0.25,1.25,0.25)
        assert len(neuron_scale_factors) == parameters['neurons']
        fc_drop_factors = np.flip(np.arange(0,1.5,0.5))
        assert len(fc_drop_factors) == parameters['fc_layers']
        #conv_drop_factors = np.flip(np.arange(0,1.25,0.25))
        conv_drop_factors = None
        #assert len(conv_drop_factors) == parameters['conv_layers']
        input_dimension_levels = np.arange(0.25,1.25,0.25)
        assert len(input_dimension_levels) == parameters['in_dim']
        input_domain_size_levels = np.arange(0.25,1.25,0.25)
        assert len(input_domain_size_levels) == parameters['in_dom_size']

        property_levels = np.arange(0,5,1)
        assert len(property_levels) == parameters['property']
        epsilon_levels = np.arange(0.4,2.4,0.4)
        assert len(epsilon_levels) == parameters['epsilon']

    else:
        assert False
    
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
    
    # TODO constraints
    '''
    lines += ['[Constraint]']
    lines += ['conv_layers=5 => in_dim!=0']
    '''
    
    lines = [x +'\n' for x in lines]

    open('./tmp/ca_config.txt','w').writelines(lines)
    cmd = 'java  -Ddoi=2 -jar lib/acts_3.2.jar ./tmp/ca_config.txt ./tmp/ca.txt > /dev/null'
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
            logger.info(f'# tests: {nb_tests}')

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

    #print(vp_configs)
    
    # gen networks

    vp_configs_ = []
    for vpc in vp_configs:
        assert len(vpc) == len(parameters)
        tmp = {}
        for i,key in enumerate(parameters):
            tmp[key] = vpc[i]
        vp_configs_ += [tmp]
    vp_configs = vp_configs_
    
    #for x in parameters:
    #    print(x)
    #print(vp_configs)

    nets = []
    for vpc in vp_configs:
        '''
        if ids_f != 1 and id_f != 1:
            print(id_f, ids_f)
            print(height,width, mean, max_value)
            print(new_height, [x*ids_f for x in mean], max_value*ids_f)
        '''

        # generate network
        
        neuron_scale_factor = neuron_scale_factors[vpc['neurons']]
        drop_fc_ids = sorted(random.sample(fc_ids, int(round(len(fc_ids)*fc_drop_factors[vpc['fc_layers']]))))
        if conv_drop_factors is not None:
            drop_conv_ids = sorted(random.sample(conv_ids, int(round(len(conv_ids)*conv_drop_factors[vpc['conv_layers']]))))
            drop_ids = drop_fc_ids + drop_conv_ids
        else:
            drop_ids = drop_fc_ids
        scale_ids = set(fc_ids + conv_ids) - set(drop_ids)

        n = Network(configs, vpc)
        dis_strats = [['drop', x, 1] for x in drop_ids]
        if neuron_scale_factor != 1:
            dis_strats += [['scale', x, 1] for x in scale_ids]
        #print(dis_strats)
        n.set_distillation_strategies(dis_strats)

        # calculate data transformaions, input demensions
        transform = n.distillation_config['distillation']['data']['transform']['student']
        height = transform['height']
        width = transform['width']
        assert height == width
        id_f = input_dimension_levels[vpc['in_dim']]

        new_height = int(np.sqrt(height*width*id_f))
        transform['height'] = new_height
        transform['width'] = new_height
        if new_height != height:
            n.scale_input = True
            n.scale_input_factor = new_height/height
        
        mean = transform['mean']
        max_value = transform['max_value']
        ids_f = input_domain_size_levels[vpc['in_dom_size']]

        transform['mean'] = [float(x*ids_f) for x in mean]
        transform['max_value'] = float(max_value*ids_f)

        if configs['image_shape_mode'] == 'NCHW':
            nb_channel = configs['image_shape'][1]
        elif configs['image_shape_mode'] == 'NHWC':
            nb_channel = configs['image_shape'][3]
        else:
            assert False
        input_shape = [nb_channel, new_height, new_height]
        
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

        nets += [n]

    logger.info(f'# NN: {len(nets)}')
    

    if args.task == 'train':
        logger.info('Training ...')
        for n in nets:
            print(n.name)
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

            prop_dir = f'{configs["props_dir"]}/{configs["network_name"]}.{input_dimension_levels[n.vpc["in_dim"]]}.{input_domain_size_levels[n.vpc["in_dom_size"]]}'

            if configs['network_name'] == 'dave':
                cmd = f'python ../r4v/tools/generate_dave_properties.py tmp/data.toml {prop_dir} -g 15'
            elif configs['network_name'] in ['mnist_conv_super', 'mnist_dense50x2']:
                cmd = f'python ../r4v/tools/generate_mnist_properties.py tmp/data.toml {prop_dir}'
            cmd += f' -e {epsilon_levels[n.vpc["epsilon"]]} -N {len(property_levels)}'
            print(cmd)
            print(n.name)
            os.system(cmd)
            exit()

    elif args.task == 'verify':
        logger.info('Verifying ...')
        verifiers = ['eran','planet','reluplex','neurify','mipverify']
        lines = []
        count = 0
        for v in verifiers:
            for n in nets:
                model = f'{n.dis_model_dir}/{n.name}.onnx'
                prop_dir = f'{configs["props_dir"]}/dave.{input_dimension_levels[n.vc[3]]}.{input_domain_size_levels[n.vc[4]]}'
                prop_levels = sorted([x for x in os.listdir(prop_dir) if '.py' in x])
                prop = prop_levels[n.vc[6]]
                cmd = f'python ../dnna/tools/resmonitor.py -T 14400 -M 64G python -m dnnv {model} {prop_dir}/{prop} --{v}'

                NODES = ['slurm1', 'slurm2', 'slurm3', 'slurm4']#, 'slurm5']
                TASK_NODE = {'slurm1':7,'slurm2':7,'slurm3':7,'slurm4':7}#,'slurm5':3}
                count += 1
                logger.info(f'Verifying ...{count}/{len(verifiers)*len(nets)}')
                vc = ''.join([str(x) for x in n.vc])

                veri_log = f'{n.config["veri_log_dir"]}/{vc}_{v}.out'
                if os.path.exists(veri_log):
                    lines = open(veri_log, 'r').readlines()

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
                
                lines = ['#!/bin/sh',
                         '#SBATCH --job-name=GB_V',
                         f'#SBATCH --output={n.config["veri_log_dir"]}/{vc}_{v}.out',
                         f'#SBATCH --error={n.config["veri_log_dir"]}/{vc}_{v}.out',
                         cmd
                ]
                lines = [x+'\n' for x in lines]
                slurm_path = os.path.join(n.config['veri_slurm_dir'],'{}_{}.slurm'.format(vc, v),)
                open(slurm_path,'w').writelines(lines)

                task = 'sbatch -w {} --reservation=dls2fc_7 {}'.format(na, slurm_path)
                os.system(task)

    elif args.task == 'ana_res':
        verifiers = ['eran','planet','reluplex','neurify','mipverify']
        results = {x:{} for x in verifiers}

        for n in nets:
            for v in verifiers:
                vpc = ''.join([str(x) for x in n.vc])
                log = f"{n.config['veri_log_dir']}/{vpc}_{v}.out"
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
                                res = lines[i].strip().split(' ')[-1]
                                v_time = float(lines[i+1].strip().split(' ')[-1])
                                results[v][vpc] = [res,v_time]
                                break

        untrained_tests = ['91003', '81003', '71003', '61003', '51002', '34030', '23001', '12031']
        print(untrained_tests)
        for v_k in results:
            assert len(results[v_k]) == len(nets)
            for vpc_k in results[v_k]:
                if vpc_k[:-2] in untrained_tests:
                    results[v_k][vpc_k] = ['untrain','']
                    #print(vpc_k, results[v_k][vpc_k])
                    
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
                elif res in ['unknown','timeout','memout','error']:
                    sums += [14400*2]
                elif res in ['untrain']:
                    pass
                else:
                    assert False
                

            par_2 = int(round(np.mean(np.array(sums))))
            
            print(f'|{v_k}|{scr}|{par_2}|')

    else:
        assert False

import os
import numpy as np
import random
import toml
import time
import pickle
import string
import datetime
import progressbar
import csv

from itertools import combinations

from .nn.onnxu import Onnxu
from .nn.layers import Dense, Conv
from .network import Network


# main benchmark generation function
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
        if key in configs['ca']['parameters']['range']:
            level_min = configs['ca']['parameters']['range'][key][0]
            level_max = configs['ca']['parameters']['range'][key][1]
            level_size = configs['ca']['parameters']['level'][key]
            if level_size == 1:
                assert level_min == level_max
                level = np.array([level_min])
            else:
                level_range = level_max - level_min
                level_step = level_range/(level_size-1)
                assert level_range > 0
                level = np.array([x*level_step+level_min for x in range(level_size)])
        elif key in configs['ca']['parameters']['enumerate']:
            level = configs['ca']['parameters']['enumerate'][key]
        
        parameters[key] = level
        # make sure all parameters are passed
        assert len(parameters[key]) == configs['ca']['parameters']['level'][key]
        
    # debug remaining layers
    possible_remaining_layers_str = "Possible remaining # of FC layers: "
    for i in range(len(parameters['fc'])):
        possible_remaining_layers_str += f"{int(round(len(fc_ids)*(1-parameters['fc'][i])))} "
    logger.debug(possible_remaining_layers_str)
    possible_remaining_layers_str = "Possible remaining # of Conv layers: "
    for i in range(len(parameters['conv'])):
        possible_remaining_layers_str += f"{int(round(len(conv_ids)*(1-parameters['conv'][i])))} "
    logger.debug(possible_remaining_layers_str)

    # print factor and levels
    logger.debug('Factor and levels:')
    for key in parameters:
        logger.debug(f'{key}: {parameters[key]}')

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

    ca_dir = f'./results/{configs["name"]}.{configs["seed"]}'
    ca_config_path = os.path.join(ca_dir,'ca_config.txt')
    ca_path = os.path.join(ca_dir,'ca.txt')

    open(ca_config_path,'w').writelines(lines)
    
    cmd = f'java  -Ddoi={strength} -jar lib/acts.jar {ca_config_path} {ca_path} > /dev/null'
    os.system(cmd)
    #os.remove(ca_config_path)

    lines = open(ca_path,'r').readlines()
    #os.remove(ca_path)

    vp_configs = []
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


    # calculate network descriptions
    nets = []
    for vpc in vp_configs:
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

        n.set_distillation_strategies(dis_strats)
        if not configs['name'] == 'acas':
            input_shape = [nb_channel, new_height, new_height]
        n.calc_order('nb_neurons', source_net_layers, input_shape)

        neurons_drop_only = np.sum(n.nb_neurons)
        neuron_scale_factor = (source_net_nb_neurons*neuron_scale_factor)/neurons_drop_only

        # check if number of kernels is than 1
        knfc = source_net.conv_kernel_and_fc_sizes
        if neuron_scale_factor != 1:
            for x in scale_ids:
                assert knfc[x] > 0
                if round(knfc[x] * neuron_scale_factor) == 0 :
                    neuron_scale_factor2 = 1/knfc[x]
                    dis_strats += [['scale', x, neuron_scale_factor2]]
                else:
                    dis_strats += [['scale', x, neuron_scale_factor]]

            n = Network(configs, vpc)

            if not configs['name'] == 'acas':
                n.distillation_config['distillation']['data']['transform']['student'] = transform
            n.set_distillation_strategies(dis_strats)
            n.calc_order('nb_neurons', source_net_layers, input_shape)
        nets += [n]

    logger.info(f'# NN: {len(nets)}')

    #  perform tasks
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
        analyze(nets, parameters, configs)

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


def train(nets, logger):
    for i in progressbar.progressbar(range(len(nets))):
        n = nets[i]
        logger.info(f'Training network {n.name} ...')
        n.train()
        logger.info(f'Training network {n.name} done.')
        #break


def gen_props(nets, parameters, configs, logger):
    #for n in nets:
    for i in progressbar.progressbar(range(len(nets))):
        n = nets[i]

        if configs['name'] == 'acas':
            prop_dir = f'{configs["props_dir"]}/{n.name}/'
            cmd =f'python ./tools/generate_acas_properties.py -d {prop_dir} -s 1'
            os.system(cmd)
            
        else:
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
                cmd =f'python ./tools/generate_dave_properties.py ./tmp/data.toml {prop_dir} -g 15 > /dev/null 2>&1'
            elif 'mcb' in configs['name'] or 'mnist' in configs['name']:
                cmd = f'python ./tools/generate_mnist_properties.py ./tmp/data.toml {prop_dir} > /dev/null 2>&1'
            else:
                raise NotImplementedError
            #nb_props = len(parameters["prop"])
            nb_props = np.max(np.array(parameters["prop"])) + 1
            cmd += f' -e {eps} -N {nb_props}'
            os.system(cmd)
            os.system('rm ./tmp/data.toml')


def verify(nets, configs, parameters, logger):
    for i in progressbar.progressbar(range(len(nets))):
        n = nets[i]
        logger.info(f'Verifying network {n.name} ...')
        n.verify(parameters,logger)
        logger.info(f'Verifying network {n.name} done.')


def analyze(nets, parameters, configs):
    verifiers = configs['verify']['verifiers']
    results = {x:{} for x in verifiers}

    # initialize result dictionary for csv output
    csv_data = []

    # collect results
    c = 0
    for n in nets:
        csv_dict = {}
        csv_dict['name'] = n.name
        for x in n.vpc:
            csv_dict[x] = parameters[x][n.vpc[x]]
        
        # find relative loss
        relative_errors = []
        lines = open(n.dis_log_path).readlines()
        for l in lines:
            if 'validation error' in l:
                relative_errors += [float(l.strip().split('=')[-1])]
        if len(relative_errors) == 0:
            print('Training failed: ', n.dis_log_path)
            min_ra = 0
        else:
            min_ra = np.min(relative_errors)
        csv_dict['relative_loss'] = min_ra
        
        for v in verifiers:
            c += 1

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

                    if 'Timeout' in l:
                        res = 'timeout'
                        break

                    if 'Out of Memory' in l:
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

                    if '[STDERR]:Error: GLP returned error 5 (GLP_EFAIL)' in l:
                        res = 'error'
                        print(res, log)
                        break

                    if "*** Error in `python':" in l:
                        res = 'error'
                        print(res, log)
                        break
                    if "nonzero(*, bool as_tuple)"in l:
                        res = 'error'
                        print(res, log)
                        break

                    if 'Cannot serialize protocol buffer of type ' in l:
                        res = 'error'
                        print(res, log)
                        break

                    if 'Bus error'in l:
                        res = 'error'
                        print(res, log)
                        break

                    '''
                    if 'No such file or directory' in l:
                        res = 'unrun'
                        print(res, log)
                        break
                    '''

                    if 'Unable to open Gurobi license file ' in l:
                        res = 'unrun'
                        print(res, log)
                        break

                # remove this
                if i == len(rlines)-1:
                    res = 'running'
                    print(res, log)
                    
            if res not in ['sat','unsat']:
                v_time = configs['verify']['time']

            assert res in ['sat','unsat','unknown','timeout','memout','error', 'unsup', 'running', 'unrun'], log
            results[v][vpc] = [res,v_time]
            csv_dict[v+'_answer'] = res
            csv_dict[v+'_time'] = v_time
        csv_data += [csv_dict]

    assert len(set([len(results[x]) for x in results.keys()])) == 1

    save_file = f'./results/{configs["name"]}_{configs["seed"]}'
    with open(f'{save_file}.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save csv dict
    
    field_names = [x for x in csv_data[0]]
    with open(f'{save_file}.csv', 'w') as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for data in csv_data:
            writer.writerow(data)

    # calculate scr and par2 scores
    scr_dict = {}
    par_2_dict = {}
    for v in verifiers:
        sums = []
        scr = 0
        for vpc_k in results[v]:
            res = results[v][vpc_k][0]
            vtime = results[v][vpc_k][1]
            if res == "running":
                print(f"running: {v} {vpc_k}")
            if res in ["sat", "unsat"]:
                scr += 1
                sums += [vtime]
            elif res in [
                    "unknown",
                    "timeout",
                    "memout",
                    "error",
                    "unsup",
                    "running",
                    "unrun",
            ]:
                sums += [configs['verify']['time'] * 2]
            elif res in ["untrain"]:
                pass
            else:
                assert False, res

        par_2 = np.mean(np.array(sums))

        if v not in scr_dict.keys():
            scr_dict[v] = [scr]
            par_2_dict[v] = [par_2]
        else:
            scr_dict[v] += [scr]
            par_2_dict[v] += [par_2]


    print('|{:>15} | {:>15} | {:>15}|'.format('Verifier','SCR','PAR-2'))
    print('|----------------|-----------------|---------------------|')
    for v in verifiers:
        print('|{:>15} | {:>15} | {:>15}|'.format(v, scr_dict[v][0], par_2_dict[v][0]))

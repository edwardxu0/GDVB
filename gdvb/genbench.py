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
import re

from itertools import combinations
from decimal import Decimal as D

from .nn.onnxu import Onnxu
from .nn.layers import Dense, Conv
from .verification_problem import VerificationProblem


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

    source_net = VerificationProblem('', configs, None)
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
                if key == 'eps':
                    level = level * configs['verify']['eps']
            else:
                level_range = D(str(level_max)) - D(str(level_min))
                level_step = D(str(level_range))/D(str(level_size-1))
                assert level_range > 0
                #level = np.array([x*level_step+level_min for x in range(level_size)])
                level = np.arange(D(str(level_min)), D(str(level_max))+level_step, D(str(level_step)))
                level = np.array(level, dtype=np.float)
                if key == 'prop':
                    level = np.array(level, dtype=np.int)
                elif key == 'eps':
                    level = np.array([D(str(x)) for x in level])
                    level = level * D(str(configs['verify']['eps']))

                assert len(level) == level_size
                
        elif key in configs['ca']['parameters']['enumerate']:
            level = configs['ca']['parameters']['enumerate'][key]
        
        parameters[key] = level
        # make sure all parameters are passed
        assert len(parameters[key]) == configs['ca']['parameters']['level'][key]

    # debug remaining layers
    if 'fc' in parameters:
        prm_str = "Possible remaining # of FC layers: "
        rml = sorted([str(int(round(len(fc_ids)*(parameters['fc'][i])))) for i in range(len(parameters['fc']))])
        prm_str += ' '.join(rml)
        logger.debug(prm_str)

    if 'conv' in parameters:
        prm_str = "Possible remaining # of Conv layers: "
        rml = sorted([str(int(round(len(conv_ids)*(parameters['conv'][i])))) for i in range(len(parameters['conv']))])
        prm_str += ' '.join(rml)
        logger.debug(prm_str)

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

    ca_config_path = os.path.join(configs['root'],'ca_config.txt')
    ca_path = os.path.join(configs['root'],'ca.txt')

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
        vp_levels = [(x,parameters[x][vpc[x]]) for x in vpc]
        vp_name = ''
        for factor, level in vp_levels:
            vp_name += f'{factor}={level}_'
        vp_name = vp_name[:-1]
        
        drop_ids = []
        if 'neu' in parameters:
            neuron_scale_factor = parameters['neu'][vpc['neu']]
        else:
            neuron_scale_factor = 1
            
        if 'fc' in parameters:
            drop_fc_ids = sorted(random.sample(fc_ids, int(round(len(fc_ids)*(1-parameters['fc'][vpc['fc']]))))) # randomly select layers to drop
            #print(int(round(len(fc_ids)*(1-parameters['fc'][vpc['fc']]))), parameters['fc'], drop_fc_ids)
            #drop_fc_ids2 = fc_ids[len(fc_ids)-int(round(len(fc_ids)*(1-parameters['fc'][vpc['fc']]))):] # select the last layer to drop
            #assert len(drop_fc_ids) == len(drop_fc_ids2)
            drop_ids += drop_fc_ids
            
        if 'conv' in parameters:
            drop_conv_ids = sorted(random.sample(conv_ids, int(round(len(conv_ids)*(1-parameters['conv'][vpc['conv']]))))) # randomly select layers to drop
            #drop_conv_ids2 = conv_ids[len(conv_ids)-int(round(len(conv_ids)*(1-parameters['conv'][vpc['conv']]))):] # select the last layer to drop
            drop_ids += drop_conv_ids
            
        if 'neu' in parameters or 'fc' in parameters or 'conv' in parameters:
            scale_ids = set(fc_ids + conv_ids) - set(drop_ids)
        else:
            scale_ids =[]

        n = VerificationProblem(vp_name, configs, vpc)
        dis_strats = [['drop', x] for x in drop_ids]

        # calculate data transformaions, input demensions
        transform = n.distillation_config['distillation']['data']['transform']['student']
        height = transform['height']
        width = transform['width']
        assert height == width

        # input demension
        if 'idm' in parameters:
            id_f = parameters['idm'][vpc['idm']]
            new_height = int(round(np.sqrt(height*width*id_f)))
            transform['height'] = new_height
            transform['width'] = new_height
            if new_height != height:
                dis_strats += [['scale_input', new_height/height]]
        else:
            new_height = height

        # input domain size
        if 'ids' in parameters:
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
        input_shape = [nb_channel, new_height, new_height]
        n.calc_order('nb_neurons', source_net_layers, input_shape)

        if 'neu' in parameters:
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

            n = VerificationProblem(vp_name, configs, vpc)

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
        train_nets = []
        train_nets_names = []
        for n in nets:
            if n.net_name not in train_nets_names:
                train_nets_names += [n.net_name]
                train_nets += [n]
        
        print(len(train_nets),len(nets))

        c = 0
        for n in train_nets:
            if n.trained():
                c += 1
        print(f'# trained nets: {c}')
        train(train_nets, logger)

    elif configs['task'] == 'gen_props':
        logger.info('Generating properties ...')
        for i in progressbar.progressbar(range(len(nets))):
            n = nets[i]

            gen_props(n, parameters, configs, logger)
            if os.path.exists(f'{configs["props_dir"]}/{n.net_name}/'):
                continue
            else:
                gen_props(n, parameters, configs, logger)


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
        logger.info(f'Training network {n.net_name} ...')
        n.train()
        logger.info(f'Training network {n.net_name} done.')
        #break


def gen_props(n, parameters, configs, logger):
    if configs['name'] == 'acas':
        prop_dir = f'{configs["props_dir"]}/{n.net_name}/'
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

        prop_dir = f'{configs["props_dir"]}/{n.net_name}/'

        if 'eps' in parameters:
            #eps = (parameters['eps']*configs['verify']['eps'])[n.vpc['eps']]
            eps = parameters['eps'][n.vpc['eps']]
        else:
            eps = configs['verify']['eps']

        if 'dave' in configs['name']:
            cmd = f'python ./tools/generate_dave_properties.py ./tmp/data.toml {prop_dir} -g 15'
        elif 'mcb' in configs['name'] or 'mnist' in configs['name'] or 'MDR' in configs['name']:
            cmd = f'python ./tools/generate_mnist_properties.py ./tmp/data.toml {prop_dir}'
        elif 'cifar' in configs['name']:
            cmd = f'python ./tools/generate_cifar10_properties.py ./tmp/data.toml {prop_dir}'
        else:
            raise NotImplementedError

        if 'prop' in parameters:
            nb_props = len(parameters["prop"])
        else:
            assert False
        cmd += f' -e {eps} -N {nb_props}'

        if configs['verify']['rm_transpose']:
            logger.warn('Removing transpose layers.')
            assert data_config['transform']['height'] == data_config['transform']['width']
            if 'dave' in configs['name']:
                nb_channels = 3
            elif 'mcb' in configs['name'] or 'mnist' in configs['name'] or 'MDR' in configs['name']:
                nb_channels = 1
            idm = data_config['transform']['height'] **2 * nb_channels
            cmd += f' -i {idm} --rm_transpose'

        cmd += ' > /dev/null 2>&1'
        print(cmd)
        os.system(cmd)
        #os.remove('./tmp/data.toml')


def verify(nets, configs, parameters, logger):
    for i in progressbar.progressbar(range(len(nets))):
        n = nets[i]

        '''
        if not n.trained:
            continue
        '''

        if not os.path.exists(n.dis_model_path):
            print(f'Network not found. Skipping ... {n.dis_model_path}')
            continue

        gen_props(n, parameters, configs, logger)
        if not os.path.exists(f'{configs["props_dir"]}/{n.net_name}/'):
            gen_props(n, parameters, configs, logger)
        else:
            prop_files = [x for x in os.listdir(f'{configs["props_dir"]}/{n.net_name}/') if '.py' in x]
            if 'eps' in parameters:
                if len(prop_files) != len(parameters['prop']) * len(parameters['eps']):
                    gen_props(n, parameters, configs, logger)
        
        logger.info(f'Verifying network {n.net_name} ... {i}/{len(nets)}')
        n.verify(parameters,logger)
        logger.info(f'Verifying network {n.net_name} done.')


def analyze(nets, parameters, configs):
    verifiers = configs['verify']['verifiers']
    results = {x:{} for x in verifiers}
    verify_epochs = configs['verify_epochs']
    iterations = 1
    if verify_epochs:
        stride = configs['verify_epochs']
        iters_to_verify = list(range(0,configs['train']['epochs']+1,stride))
        iterations = len(iters_to_verify)


    for i_epoch in range(iterations):
        # initialize result dictionary for csv output
        csv_data = []
        train_loss = {}

        # collect results
        c = 0
        for n in nets:
            csv_dict = {}
            csv_dict['name'] = n.net_name
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
                if verify_epochs:
                    if iters_to_verify[i_epoch] == 0:
                        min_ra = 0
                    else:
                        min_ra = ([0]+relative_errors)[iters_to_verify[i_epoch]]
                else:
                    min_ra = np.min(relative_errors)
            csv_dict['relative_loss'] = round(min_ra, 4)
            train_loss[n.net_name] = relative_errors
            

            for v in verifiers:
                c += 1

                vpc = ':'.join([str(n.vpc[x]) for x in n.vpc])
                if verify_epochs:
                    log = os.path.join(n.config["veri_log_dir"] ,f'{n.vp_name}_T={configs["verify"]["time"]}_M={configs["verify"]["memory"]}:{v}.{iters_to_verify[i_epoch]}.out')
                else:
                    log = os.path.join(n.config["veri_log_dir"] ,f'{n.vp_name}_T={configs["verify"]["time"]}_M={configs["verify"]["memory"]}:{v}.out')
                if not os.path.exists(log):
                    res = 'unrun'
                    print('unrun', log)
                    v_time = -1
                else:
                    nb_check_lines = 300
                    rlines = list(reversed(open(log,'r').readlines()))[:nb_check_lines]
                    rlines2 = list(reversed(open(log[:-3]+'err','r').readlines()))[:nb_check_lines]
                    rlines = rlines2 + rlines

                    res = None
                    v_time = None
                    for i,l in enumerate(rlines):

                        if re.match(r'INFO*', l):
                            continue
                        
                        if re.search('Timeout', l):
                            res = 'timeout'
                            v_time = configs['verify']['time']
                            break
                        
                        if re.search('Out of Memory', l):
                            res = 'memout'
                            for l in rlines:
                                if re.search('Duration', l):
                                    v_time = float(l.split(' ')[9][:-2])
                                    break
                            break
                        
                        if re.search('RuntimeError: view size is not compatible',l):
                            res = 'error'
                            v_time = configs['verify']['time']
                            break
                            
                        
                        if re.search(' result: ', l):
                            error_patterns = ['PlanetError',
                                              'ReluplexError',
                                              'ReluplexTranslatorError',
                                              'ERANError',
                                              'MIPVerifyTranslatorError',
                                              'NeurifyError',
                                              'NnenumError',
                                              'MarabouError',
                                              'VerinetError',
                                              'MIPVerifyError'
                                              ]
                            if any(re.search(x,l) for x in error_patterns):
                                res = 'error'
                            elif re.search('Return code: -11', l):
                                res = 'memout'
                            else:
                                res = l.strip().split(' ')[-1]
                            v_time = float(rlines[i-1].strip().split(' ')[-1])
                            break


                        # exceptions that DNNV didn't catch
                        exception_patterns = ["Aborted         "]
                        if any(re.search(x,l) for x in exception_patterns):
                            res = 'exception'
                            for l in rlines:
                                if re.search('Duration', l):
                                    v_time = float(l.split(' ')[9][:-2])
                                    break
                            break

                        # failed jobs that are likely due to server error
                        rerun_patterns = ['CANCELLED AT',
                                          'Unable to open Gurobi license file',
                                          'cannot reshape array of size 0 into shape',
                                          'property_node = module.body[-1]',
                                          'slurmstepd: error: get_exit_code',
                                          'Cannot load file containing pickled data',
                                          'IndexError: list index out of range',
                                          'gurobipy.GurobiError: No Gurobi license',
                                          'gurobipy.GurobiError: License expired ',
                                          'Cannot allocate memory',
                                          'Disk quota exceeded',
                                          'ValueError: Unknown arguments: --',
                                          '--- Logging error ---'
                                          ]
                        if any(re.search(x, l) for x in rerun_patterns):
                            res = 'rerun'
                            v_time = -1
                            print(res, log)
                            break
                
                if not res and (i+1 in [nb_check_lines, len(rlines)] or len(rlines)==0):
                    res = 'undetermined'
                    v_time = -1
                    print(res,log)

                #if v == 'nnenum':
                #    print(res, v_time)
                        
                assert res, res
                assert v_time, f'{v_time} : {log}'
                if res not in ['sat','unsat','unknown','error','timeout','memout','exception','rerun','torun','unrun','undetermined']:
                    assert False, f'{res}:{log}'

                results[v][vpc] = [res,v_time]
                csv_dict[v+'_answer'] = res
                csv_dict[v+'_time'] = round(v_time,2)
            csv_data += [csv_dict]

        assert len(set([len(results[x]) for x in results.keys()])) == 1

        save_file = f'{configs["root"]}/{configs["name"]}_{configs["seed"]}'

        train_loss_lines = []
        # save training losses as csv
        for x in train_loss:
            line = f'{x},'
            for xx in train_loss[x]:
                line = line+f'{xx}, '
            train_loss_lines += [line]
        train_loss_lines = [x+'\n' for x in train_loss_lines]

        with open(f'{save_file}_train_loss.csv', 'w') as handle:
            handle.writelines(train_loss_lines)


        # save verification results as pickle
        if verify_epochs:
            save_file += f'_{iters_to_verify[i_epoch]}'
        with open(f'{save_file}.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save verification results as csv
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
                if res in ["sat", "unsat"]:
                    scr += 1
                    sums += [vtime]
                elif res in ['unknown','error','timeout','memout','exception','rerun','torun','unrun','undetermined']:
                    sums += [configs['verify']['time'] * 2]
                else:
                    assert False

            par_2 = np.mean(np.array(sums))

            if v not in scr_dict.keys():
                scr_dict[v] = [scr]
                par_2_dict[v] = [par_2]
            else:
                scr_dict[v] += [scr]
                par_2_dict[v] += [par_2]


        print('|{:>15} | {:>15} | {:>15}|'.format('Verifier','SCR','PAR-2'))
        print('|----------------|-----------------|----------------|')
        for v in verifiers:
            print('|{:>15} | {:>15} | {:>15.2f}|'.format(v, scr_dict[v][0], round(par_2_dict[v][0],2)))

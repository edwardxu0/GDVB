import os
import sys
import copy
import numpy as np
import random
import toml
import time
import logging
logging.basicConfig(stream=sys.stdout,
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

import gb4v.verify.neurify

from nn.layers import Dense, Conv, Transpose, Flatten

NODES = ['slurm1', 'slurm2', 'slurm3', 'slurm4', 'slurm5']
TASK_NODE = {'slurm1':7,'slurm2':7,'slurm3':7,'slurm4':7,'slurm5':3}


class Network():
    def __init__(self, config, vpc):
        self.name = config['name']
        if vpc is not None:
            self.name += '.'+''.join([str(vpc[x]) for x in list(vpc)[:-2]])            
        self.vpc = vpc
        self.config = config
        self.distillation_configled = False
        self.distilled = False
        self.tested = False
        self.verified = False
        self.processed = False
        self.relative_acc = None
        self.best_iter = None
        self.d_time = None
        self.score_veri = 0
        self.scale_input = False

        source_dis_config = open(self.config['dnn']['r4v_config'],'r').read()
        self.distillation_config = toml.loads(source_dis_config)

        
    def set_distillation_strategies(self, dis_strats):
        self.dis_strats = dis_strats

        drop_ids = []
        scale_ids_factors = []
        for ds in dis_strats:
            if ds[0] == 'drop':
                drop_ids += [ds[1]]
            elif ds[0] == 'scale':
                scale_ids_factors += [[ds[1],ds[2]]]
            elif ds[0] == 'scale_input':
                self.scale_input = True
                self.scale_input_factor = ds[1]
            else:
                assert False, 'Unkown strategy'+ ds

        strats = [x[0] for x in dis_strats]
        if 'scale' in strats:
            self.name+=f'_{dis_strats[strats.index("scale")][2]:.3f}'
        else:
            self.name+=f'_{1:.3f}'
        
        '''
        if drop_ids:
            self.name += '.D.' + '.'.join([str(x) for x in sorted(drop_ids)])
        if scale_ids_factors:
            self.name += '.S'
            for i in range(len(scale_ids_factors)):
                self.name += '.{}_{}_'.format(scale_ids_factors[i][0], str(scale_ids_factors[i][1])[:5])
        '''
        self.drop_ids = drop_ids
        self.scale_ids_factors = scale_ids_factors

        #self.distillation_config['distillation']['parameters']['epochs'] = self.config['proxy_dis_epochs']

        self.dis_config_path = os.path.join(self.config['dis_config_dir'], self.name + '.toml')
        #self.dis_model_dir = os.path.join(self.config['dis_model_dir'], self.name)
        self.dis_model_path = os.path.join(self.config['dis_model_dir'], self.name +'.onnx')
        self.dis_log_path = os.path.join(self.config['dis_log_dir'], self.name + '.out')
        self.dis_slurm_path = os.path.join(self.config['dis_slurm_dir'], self.name + '.slurm')

        # TODO: use better toml api
        #self.distillation_config['distillation']['student'] = {}
        #self.distillation_config['distillation']['student']['path'] = self.dis_model_dir


    def calc_order(self, order_by, orig_layers, input_shape):
        if order_by == 'nb_neurons':
            self.order_by = 'nb_neurons'
            self.layers = []
            self.nb_neurons = []
            self.remaining_layer_ids = []

            for i in range(len(orig_layers)):
                if i not in self.drop_ids:
                    if self.layers == []:
                        in_shape = input_shape
                    else:
                        in_shape = self.layers[-1].out_shape

                    ol = orig_layers[i]
                    if ol.type == 'FC':
                        size = ol.size
                        for x in self.scale_ids_factors:
                            if x[0] == i:
                                #size = int(size * x[1])
                                size = int(round(size * x[1]))
                                break
                        l = Dense(size, None, None, in_shape)
                        self.nb_neurons += [np.prod(l.out_shape)]
                    elif ol.type == 'Conv':
                        size = ol.size
                        for x in self.scale_ids_factors:
                            if x[0] == i:
                                #size = int(size * x[1])
                                size = int(round(size * x[1]))
                                break
                        l = Conv(size, None, None, ol.kernel_size, ol.stride, ol.padding, in_shape)
                        self.nb_neurons += [np.prod(l.out_shape)]
                    elif ol.type == 'Transpose':
                        l = Transpose(ol.order, in_shape)
                        self.nb_neurons += [0]
                    elif ol.type == 'Flatten':
                        l = Flatten(in_shape)
                        self.nb_neurons += [0]
                    else:
                        assert False
                    # print(l)
                    self.layers += [l]
                    self.remaining_layer_ids += [i]
            # print(np.sum(np.array(self.nb_neurons)))
        else:
            assert False


    # override comparators
    def __gt__(self, other):
        if self.order_by == 'nb_neurons' and other.order_by == 'nb_neurons':
            return np.sum(self.nb_neurons) > np.sum(other.nb_neurons)
        else:
            assert False


    def distill(self):
        logging.info('Distilling network: ' + self.name)

        if os.path.exists(self.dis_log_path):
            lines = open(self.dis_log_path).readlines()
            successful = True
            for l in lines:
                if 'Traceback' in l:
                    successful = False
            if successful:
                print('done')
                return
            else:
                print(f'error in {self.dis_log_path}, retraining')

        formatted_data = toml.dumps(self.distillation_config)
        with open(self.dis_config_path, 'w') as f:
            f.write(formatted_data)

        # TODO: use better toml api
        lines = ['']
        if self.drop_ids:
            lines +=['[[distillation.strategies.drop_layer]]']
            lines +=['layer_id=['+', '.join([str(x) for x in self.drop_ids])+']']
            lines +=['']
        if self.scale_ids_factors:
            lines +=['[[distillation.strategies.scale_layer]]']
            lines +=['layer_id=['+', '.join([str(x[0]) for x in self.scale_ids_factors])+']']
            lines +=['factor=[{}]'.format(', '.join([str(x[1]) for x in self.scale_ids_factors]))]
            lines +=['']
        if self.scale_input:
            lines +=['[[distillation.strategies.scale_input]]']
            lines +=[f'factor={self.scale_input_factor}\n']

        lines += ['[distillation.student]']
        lines += ['path="'+self.dis_model_path+'"']
        lines = [x+'\n' for x in lines]

        with open(self.dis_config_path, 'a') as f:
            for l in lines:
                f.write(l)
        '''
        if not os.path.exists(self.dis_model_dir):
            os.mkdir(self.dis_model_dir)
        '''

        slurm_lines = ['#!/bin/sh',
                       '#SBATCH --job-name=GB_D',
                       '#SBATCH --error="{}"'.format(self.dis_log_path),
                       '#SBATCH --output="{}"'.format(self.dis_log_path),
                       '#SBATCH --partition=gpu',
                       '#SBATCH --gres=gpu:1',
                       'cat /proc/sys/kernel/hostname',
                       'python -m r4v distill {} --debug'.format(self.dis_config_path)
        ]
        slurm_lines = [x+'\n' for x in slurm_lines]
        open(self.dis_slurm_path, 'w').writelines(slurm_lines)
        
        '''
        tmp_file = './tmp/'+str(np.random.uniform(low=0, high=10000, size=(1))[0])
        cmd = 'squeue -u dx3yy > ' + tmp_file
        os.system(cmd)
        time.sleep(0)
        sq_lines = open(tmp_file, 'r').readlines()[1:]
        ccc = 0
        for l in sq_lines:
            if 'NAS-BS_D' in l or 'NAS-BS_T' in l:
                ccc += 1
        
        if ccc < 6:
            run_node = ' -w ristretto01 --reservation=dx3yy_15 '
        elif ccc < 13:
            run_node = ' -w ristretto02 --reservation=dx3yy_15 '
        else:
            run_node = ''
        run_node = ''

        #cmd = 'sbatch {} {}'.format(run_node, self.dis_slurm_path)
        '''
        cmd = 'sbatch --reservation=dx3yy_9 --exclude=artemis4,artemis5,artemis6,artemis7 {}'.format(self.dis_slurm_path)
        os.system(cmd)


    def dis_monitor(self):
        print('\n--- INFO --- Checking distillation status: ' + self.name)
        while True:
            time.sleep(1)
            done_nets = os.listdir(self.config['dis_done_dir'])
            #print("distilled networks:")
            #print(done_nets)
            if self.name+'.done' in done_nets:
                self.distilled = True
                break
        

        with open(self.dis_log_path, 'r') as f:
            lines = f.readlines()
            from datetime import datetime
            d_str = lines[0].split(' ')[4]
            d_int = [int(x) for x in d_str.split('-')]
            t_str = lines[0].split(' ')[5]
            t_int = [int(x) for x in t_str.split(':')[:-1]]
            t_int += [int(t_str.split(':')[-1][:-4])]
            d1 = datetime(d_int[0],d_int[1],d_int[2],t_int[0],t_int[1],t_int[2])

            canceled = False
            for l in reversed(lines):
                if 'CANCELLED' in l:
                    canceled = True

            if not canceled:
                for l in lines:
                    if 'DEBUG    2019-' in l:
                        d_str = l.split(' ')[4]
                        d_int = [int(x) for x in d_str.split('-')]
                        t_str = l.split(' ')[5]
                        t_int = [int(x) for x in t_str.split(':')[:-1]]
                        t_int += [int(t_str.split(':')[-1][:-4])]
                        d2 = datetime(d_int[0],d_int[1],d_int[2],t_int[0],t_int[1],t_int[2])
                duration = (d2 - d1).total_seconds()
                print(duration)
            else:
                duration = 0

            self.d_time = duration


    def test(self):
        print('\n--- INFO --- Testing network: ' + self.name)

        done_nets = os.listdir(self.config['test_done_dir'])
        if self.name+'.done' in done_nets:
            return

        slurm_lines = ['#!/bin/sh',
                       '#SBATCH --job-name=NAS-BS_T',
                       '#SBATCH --partition=gpu',
                       '#SBATCH --error="{}"'.format(self.test_log_path),
                       '#SBATCH --output="{}"'.format(self.test_log_path),
                       '#SBATCH --gres=gpu:1',
                       '',
                       'time (',
                       'python /p/d4v/dx3yy/DNNVeri/dnnvbs/cegsdl/tools/measure_performance.py {} configs/udacity-driving.100.valid.toml --input_shape 1 3 100 100 --loss mse --cuda --teacher /p/d4v/dx3yy/DNNVeri/dnnvbs/cegsdl/networks/dave/model.onnx --teacher_input_shape 1 100 100 3 --teacher_input_format NHWC'.format(self.dis_model_dir),
                       ')',
                       'touch {}'.format(self.test_done_path)]
        slurm_lines = [x+'\n' for x in slurm_lines]
        open(self.test_slurm_path, 'w').writelines(slurm_lines)
        tmp_file = './tmp/'+str(np.random.uniform(low=0, high=10000, size=(1))[0])
        tmp_file = 'sq.txt'
        cmd = 'squeue -u dx3yy > ' + tmp_file
        os.system(cmd)
        time.sleep(5)
        sq_lines = open(tmp_file, 'r').readlines()[1:]
        ccc = 0
        for l in sq_lines:
            if 'NAS-BS_D' in l or 'NAS-BS_T' in l:
                ccc += 1
        if ccc < 6:
            run_node = ' -w ristretto01 --reservation=dx3yy_15 '
        elif ccc < 13:
            run_node = ' -w ristretto02 --reservation=dx3yy_15 '
        else:
            run_node = ''

        cmd = 'sbatch {} {}'.format(run_node, self.test_slurm_path)
        #cmd = 'sbatch -w ristretto01 --reservation=dx3yy_15 {}'.format(self.test_slurm_path)
        #cmd = 'sbatch --exclude=artemis4,artemis5,artemis6,artemis7,lynx05,lynx06 {}'.format(self.test_slurm_path)
        os.system(cmd)


    def test_monitor(self):
        print('\n--- INFO --- Waiting and analyzing network test result: ' + self.name)

        while True:
            time.sleep(1)
            done_nets = os.listdir(self.config['test_done_dir'])
            # print("tested networks:")
            # print(done_nets)
            if self.name+'.done' in done_nets:
                self.tested = True
                break

        # Parser test results
        lines = open(self.test_log_path, 'r').readlines()

        #self.relative_acc = float(lines[-2].strip().split(' ')[-1])
        #self.best_iter = int(lines[-3].strip().split(' ')[-1])
        self.relative_acc = float(lines[-6].strip().split(' ')[-1])
        self.best_iter = int(lines[-7].strip().split(' ')[-1])

        self.best_model_name = self.name+'.iter.'+str(self.best_iter)
        self.best_model_path = os.path.join(self.dis_model_dir, self.best_model_name+'.onnx')

        print('Test relative accuracy: ', self.relative_acc)
        print('Best iteration: ', self.best_iter)
        

    def transform(self):
        print('\n--- INFO --- Transforming network: ' + self.name)
        for v in self.config['verifiers']:
            if v == 'neurify':
                if os.path.exists('{}/{}.{}'.format(self.config['veri_net_dir'], self.best_model_name,v)):
                    continue
                else:
                    cmd = '/p/d4v/dx3yy/DNNVeri/dnnvbs/tools/main.py -v neurify  -om {} Convert'.format(self.best_model_path)
                    os.system(cmd)
                    cmd = 'cp /p/d4v/dx3yy/DNNVeri/dnnvbs/results/networks/{}/{}.nnet {}/{}.{}'.format(v,self.best_model_name, self.config['veri_net_dir'], self.best_model_name,v)
                    os.system(cmd)
            else:
                assert False


    def verify(self):
        print('\n--- INFO --- Verifying network: ' + self.name)
        prop_bounds = {}
        lines = open(os.path.join(self.config['prop_dir'],self.config['network_name'],'properties.csv'),'r').readlines()[1:]
        for l in lines:
            toks = l.strip().split(',')
            prop_bounds[toks[0]] = [toks[-2], toks[-1]]
        all_props = os.listdir(os.path.join(self.config['prop_dir'],self.config['network_name']))

        veri_jobs = []
        ccc = 0
        for v in self.config['verifiers']:
            props = [x for x in all_props if v in x]
            # Verification Proxy, sample properties
            random.seed(self.config['seed'])
            props_sampled = random.sample(props, self.config['proxy_nb_props'])
            
            for p in props_sampled:

                p_name = os.path.splitext(p)[0]
                veri_jobs += ['{}_{}_{}'.format(self.best_model_name, p_name, v)]
                [lb, ub] = prop_bounds[p_name]

                if os.path.exists(('{}/{}_{}_{}.done'.format(self.config['veri_done_dir'], self.best_model_name, p_name, v))):
                    #print('YES')
                    continue

                while(True):
                    node_avl_flag = False
                    task = 'squeue -u dx3yy > squeue_results.txt'
                    os.system(task)
                    time.sleep(5)
                    sq_lines = open('squeue_results.txt', 'r').readlines()[1:]
                    nodes_avl = {}
                    for n in NODES:
                        nodes_avl[n] = 0

                    nodenodavil_flag = False
                    for l in sq_lines:
                        if 'ReqNodeNotAvail' in l and 'dx3yy' in l:
                            nodenodavil_flag = True
                            break
                    
                        if ' R ' in l and l != '':
                            node = l.strip().split(' ')[-1]
                            if node in NODES:
                                nodes_avl[node] += 1
                    '''
                    if nodenodavil_flag == True:
                        print('node unavialiable. waiting ...')
                        continue
                    '''

                    #print(nodes_avl)
            
                    for na in nodes_avl:
                        if nodes_avl[na] < TASK_NODE[na]:
                            node_avl_flag = True
                            break
                    if node_avl_flag:
                        break


                lines = ['#!/bin/sh',
                         '#SBATCH --job-name=NAS-BS_V',
                         '#SBATCH --output={}/{}_{}_{}.out'.format(self.config['veri_log_dir'], self.best_model_name, p_name, v),
                         '#SBATCH --error={}/{}_{}_{}.out'.format(self.config['veri_log_dir'], self.best_model_name, p_name, v),
                         'time (',
                         'stdbuf -o0 -e0 python /p/d4v/dx3yy/DNNVeri/dnnvbs/cegsdl/tools/resmonitor.py -M {}G -T {} /p/d4v/dx3yy/DNNVeri/dnnvbs/verifiers/Neurify/dave2/network_test 500 {} {} {} {}'
                         .format(self.config['veri_mem'],
                                 self.config['veri_time'],
                                 os.path.join(self.config['veri_net_dir'],self.best_model_name+'.'+v),
                                 os.path.join(self.config['prop_dir'],self.config['network_name'], p),
                                 lb, ub),
                         ')',
                         'touch {}/{}_{}_{}.done'.format(self.config['veri_done_dir'], self.best_model_name, p_name, v),]
                lines = [x+'\n' for x in lines]
                slurm_path = os.path.join(self.config['veri_slurm_dir'],'{}_{}_{}.slurm'.format(self.best_model_name, p_name, v),)
                open(slurm_path,'w').writelines(lines)

                task = 'sbatch -w {} --reservation=dx3yy_15 {}'.format(na, slurm_path)
                os.system(task)


    def veri_monitor(self):
        print('\n--- INFO --- Waiting and analyzing verified network: ' + self.name)
        all_props = os.listdir(os.path.join(self.config['prop_dir'],self.config['network_name']))
        self.veri_res = {}
        veri_jobs = []
        for v in self.config['verifiers']:
            props = [x for x in all_props if v in x]
            random.seed(self.config['seed'])
            props_sampled = random.sample(props, self.config['proxy_nb_props'])
            for p in props_sampled:
                p_name = os.path.splitext(p)[0]
                veri_jobs += ['{}_{}_{}'.format(self.best_model_name, p_name, v)]

        while veri_jobs != []:
            time.sleep(1)
            done_veri = os.listdir(self.config['veri_done_dir'])
            
            #print("veri jobs:")
            #print(veri_jobs)
            #print("verified networks:")
            #print(done_veri)

            i = 0
            while i < len(veri_jobs):
                vj = veri_jobs[i]
                if vj+'.done' in done_veri:
                    if v == 'neurify':
                        [res, v_time]= verify.neurify.analyze_result(os.path.join(self.config['veri_log_dir'], vj+'.out'))
                    else:
                        assert False

                    p_name = vj.split('_')[1]
                    if v not in self.veri_res.keys():
                        self.veri_res[v] = [[p_name, res, v_time]]
                    else:
                        self.veri_res[v] += [[p_name, res, v_time]]
                    del veri_jobs[i]
                    print('verification result:', v, p_name, res, v_time)
                else:
                    i += 1

            # Stop when all verification jobs are done for one verification tool
            finished_verifier = None
            for key in self.veri_res.keys():
                if len(self.veri_res[key]) == self.config['proxy_nb_props']:
                    finished_verifier = key
            if finished_verifier is not None:
                #print('Verifier finished:', finished_verifier)
                break


    def input_too_small(self):

        conv_ids = []
        for i,l in enumerate(self.layers):
            if isinstance(l, Conv):
                conv_ids += [i]
        assert set(conv_ids) == set(range(len(conv_ids)))
        conv_ids.reverse()

        if not conv_ids:
            return False
        
        #print(conv_ids)
        return False

        
    def untrainable(self, parameter_limit):
        self.nb_parameters = 0
        for i in range(len(self.layers)):
            l = self.layers[i]
            if isinstance(l, Conv):
                self.nb_parameters += (l.kernel_size**2 + 1)*l.size
            elif isinstance(l, Dense):
                ins = np.prod(np.array(l.in_shape))
                outs = np.prod(np.array(l.out_shape))
                self.nb_parameters += ins*outs
        return self.nb_parameters > parameter_limit

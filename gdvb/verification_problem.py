import os
import sys
import numpy as np
import random
import toml
import time
import logging
import string

from decimal import Decimal as D

from .nn.layers import Dense, Conv, Transpose, Flatten
from .dispatcher import Task


class VerificationProblem():
    def __init__(self, vp_name, config, vpc):
        self.vp_name = vp_name
        self.net_name = ''
        tokens = self.vp_name.split('_')
        for t in tokens:
            if not t.startswith('eps=') and not t.startswith('prop='):
                self.net_name += f'{t}_'
        self.net_name = self.net_name[:-1]

        self.name = config['name']
        if vpc is not None:
            if not config['name'] == 'acas':
                temp_vpc = [x for x in vpc if x not in ['eps','prop']]
            else:
                temp_vpc = [vpc[x] for x in vpc if x not in ['prop']]
            self.name += '.'+':'.join([str(vpc[x]) for x in temp_vpc])
        self.vpc = vpc

        self.config = config
        
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
            self.net_name += f'_SF={dis_strats[strats.index("scale")][2]:.3f}'
            self.vp_name += f'_SF={dis_strats[strats.index("scale")][2]:.3f}'
        else:
            self.name += f'_{1:.3f}'
            self.net_name += f'_{1:.3f}'
            self.vp_name += f'_{1:.3f}'
        
        self.drop_ids = drop_ids
        self.scale_ids_factors = scale_ids_factors

        '''
        self.distillation_config['distillation']['parameters']['epochs'] = self.config['train']['epochs']
        self.dis_config_path = os.path.join(self.config['dis_config_dir'], self.name + '.toml')
        self.dis_model_path = os.path.join(self.config['dis_model_dir'], self.name +'.onnx')
        self.dis_log_path = os.path.join(self.config['dis_log_dir'], self.name + '.out')
        self.dis_slurm_path = os.path.join(self.config['dis_slurm_dir'], self.name + '.slurm')

        self.dis_config_path2 = os.path.join(self.config['dis_config_dir'], self.net_name + '.toml')
        self.dis_model_path2 = os.path.join(self.config['dis_model_dir'], self.net_name +'.onnx')
        self.dis_log_path2 = os.path.join(self.config['dis_log_dir'], self.net_name + '.out')
        self.dis_slurm_path2 = os.path.join(self.config['dis_slurm_dir'], self.net_name + '.slurm')
        '''
        
        self.dis_config_path = os.path.join(self.config['dis_config_dir'], self.net_name + '.toml')
        self.dis_model_path = os.path.join(self.config['dis_model_dir'], self.net_name +'.onnx')
        self.dis_log_path = os.path.join(self.config['dis_log_dir'], self.net_name + '.out')
        self.dis_slurm_path = os.path.join(self.config['dis_slurm_dir'], self.net_name + '.slurm')

        
    # calculate neurons/layers
    def calc_order(self, order_by, orig_layers, input_shape):
        if order_by == 'nb_neurons':
            self.order_by = 'nb_neurons'
            self.layers = []
            self.nb_neurons = []
            self.remaining_layer_ids = []
            self.conv_kernel_and_fc_sizes = []

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
                                size = int(round(size * x[1]))
                                break
                        self.conv_kernel_and_fc_sizes += [size]
                        l = Dense(size, None, None, in_shape)
                        self.nb_neurons += [np.prod(l.out_shape)]
                    elif ol.type == 'Conv':
                        size = ol.size
                        for x in self.scale_ids_factors:
                            if x[0] == i:
                                size = int(round(size * x[1]))
                                break
                        self.conv_kernel_and_fc_sizes += [size]
                        l = Conv(size, None, None, ol.kernel_size, ol.stride, ol.padding, in_shape)
                        self.nb_neurons += [np.prod(l.out_shape)]
                    elif ol.type == 'Transpose':
                        l = Transpose(ol.order, in_shape)
                        self.conv_kernel_and_fc_sizes += [-1]
                        self.nb_neurons += [0]
                    elif ol.type == 'Flatten':
                        l = Flatten(in_shape)
                        self.conv_kernel_and_fc_sizes += [-1]
                        self.nb_neurons += [0]
                    else:
                        assert False
                    self.layers += [l]
                    self.remaining_layer_ids += [i]
        else:
            assert False


    # override comparators
    def __gt__(self, other):
        if self.order_by == 'nb_neurons' and other.order_by == 'nb_neurons':
            return np.sum(self.nb_neurons) > np.sum(other.nb_neurons)
        else:
            assert False
    
    # train network
    def train(self):

        '''
        print(self.dis_config_path)
        assert os.path.exists(self.dis_config_path), self.dis_config_path
        cmd = f'cp {self.dis_config_path} {self.dis_config_path2}'
        print(cmd)
        os.system(cmd)
        #assert os.path.exists(self.dis_model_path), self.dis_model_path
        cmd = f'cp {self.dis_model_path} {self.dis_model_path2}'
        print(cmd)
        os.system(cmd)
        assert os.path.exists(self.dis_log_path), self.dis_log_path
        cmd = f'cp {self.dis_log_path} {self.dis_log_path2}'
        print(cmd)
        os.system(cmd)
        assert os.path.exists(self.dis_slurm_path), self.dis_slurm_path
        cmd = f'cp {self.dis_slurm_path} {self.dis_slurm_path2}'
        print(cmd)
        os.system(cmd)
        '''
        
        if not self.config['override']:
            if os.path.exists(self.dis_log_path):
                print('NN already trained. Skipping ... ', self.dis_log_path)
                return
        
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

        open(self.dis_config_path, 'a').writelines(lines)

        cmds = ['./scripts/run_r4v.sh distill {} --debug'.format(self.dis_config_path)]
        task = Task(cmds,
                    self.config['train']['dispatch'],
                    "GDVB_Train",
                    self.dis_log_path,
                    self.dis_slurm_path
        )
        task.run()

    def trained(self):
        last_iter_model_path = self.dis_model_path[:-4]+'iter.10.onnx'
        trained = os.path.exists(last_iter_model_path)

        if os.path.exists(self.dis_log_path) and not trained:
            print(f'removing {self.dis_log_path}')
            #os.remove(self.dis_log_path)

        '''
        if os.path.exists(self.dis_log_path):
            lines = open(self.dis_log_path, 'r').readlines()[-10:]
            for l in lines:
                if 'Process finished successfully' in l:
                    trained = True
                    break
        '''
 
        return trained

    # verify network
    def verify(self, parameters, logger):
        configs = self.config
        verifiers = configs['verify']['verifiers']
        time_limit = configs['verify']['time']
        memory_limit = configs['verify']['memory']

        count = 0
        iterations = 1
        verify_epochs = self.config['verify_epochs']
        if verify_epochs:
            stride = verify_epochs
            iters_to_verify = list(range(0,self.config['train']['epochs']+1,stride))
            iterations = len(iters_to_verify)

        for i in range(iterations):
            for v in verifiers:
                count += 1
                logger.info(f'Verifying with {v} {count}/{len(verifiers)}')
                vpc = ':'.join([str(self.vpc[x]) for x in self.vpc])
                
                if 'eran' in v:
                    verifier_cmd = f'--eran --eran.domain {v.split("_")[1]}'
                elif v == 'bab_sb':
                    verifier_cmd = '--bab --bab.smart_branching True'
                elif v == 'dnnf':
                    verifier_cmd = ''
                else:
                    verifier_cmd = f'--{v.replace("_ins","")}'

                net_path = self.dis_model_path
                self.veri_log_path = os.path.join(self.config["veri_log_dir"] ,f'{self.vp_name}_T={configs["verify"]["time"]}_M={configs["verify"]["memory"]}:{v}.out')
                self.slurm_script_path = os.path.join(self.config['veri_slurm_dir'],f'{self.vp_name}_T={configs["verify"]["time"]}_M={configs["verify"]["memory"]}:{v}.slurm')
                
                '''
                veri_log_path = f'{self.config["veri_log_dir"]}/{vpc}_{v}.out'
                slurm_script_path = os.path.join(self.config['veri_slurm_dir'],f'{vpc}_{v}.slurm')

                veri_log_path2 = os.path.join(self.config["veri_log_dir"] ,f'{self.vp_name}_T={configs["verify"]["time"]}_M={configs["verify"]["memory"]}:{v}.out')
                slurm_script_path2 = os.path.join(self.config['veri_slurm_dir'],f'{self.vp_name}_T={configs["verify"]["time"]}_M={configs["verify"]["memory"]}:{v}.slurm')

                os.path.exists(veri_log_path)
                cmd = f'cp {veri_log_path} {veri_log_path2}'
                print(cmd)
                os.system(cmd)
                os.path.exists(slurm_script_path)
                cmd = f'cp {slurm_script_path} {slurm_script_path2}'
                print(cmd)
                os.system(cmd)
                '''
                
                if verify_epochs:
                    net_path, ext = os.path.splitext(net_path)
                    net_path = f'{net_path}.iter.{iters_to_verify[i]}{ext}'
                    self.veri_log_path, ext = os.path.splitext(self.veri_log_path)
                    self.veri_log_path = f'{self.veri_log_path}.{iters_to_verify[i]}{ext}'
                    slurm_script_path, ext = os.path.splitext(slurm_script_path)
                    slurm_script_path = f'{slurm_script_path}.{iters_to_verify[i]}{ext}'

                if not configs['override']:
                    if os.path.exists(self.veri_log_path):
                        print('Instance already verified. Skipping ... ', self.veri_log_path)
                        continue

                if 'eps' in parameters:
                    eps = parameters['eps'][self.vpc['eps']]
                else:
                    eps = configs['verify']['eps']

                prop_dir = f'{configs["props_dir"]}/{self.net_name}/'
                prop_levels = sorted([x for x in os.listdir(prop_dir) if '.py' in x])
                prop_levels = [x for x in prop_levels if f'.{float(str(eps))}.py' in x]
                assert len(prop_levels) == len(parameters['prop']), f"{len(prop_levels)}, {len(parameters['prop'])}"
                
                prop = prop_levels[self.vpc['prop']]

                if v == 'dnnf':
                    executor = 'run_dnnf.sh'
                elif v.endswith('_ins'):
                    executor = 'run_dnnv_ins.sh'
                else:
                    executor = 'run_dnnv.sh'
                cmd = f'python -W ignore ./lib/DNNV/tools/resmonitor.py -T {time_limit} -M {memory_limit}'
                cmd += f' ./scripts/{executor} {prop_dir}/{prop} --network N {net_path} {verifier_cmd}'
                if self.config['verify']['debug']:
                    cmd += ' --debug'
                cmds = [cmd]
                
                task = Task(cmds,
                            self.config['verify']['dispatch'],
                            "GDVB_Verify",
                            self.veri_log_path,
                            self.slurm_script_path
                )
                task.run()

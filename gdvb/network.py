import os
import sys
import numpy as np
import random
import toml
import time
import logging
import string

from .nn.layers import Dense, Conv, Transpose, Flatten
from .dispatcher import Task


class Network():
    def __init__(self, config, vpc):
        self.name = config['name']
        if vpc is not None:
            if not config['name'] == 'acas':
                self.name += '.'+''.join([str(vpc[x]) for x in list(vpc)[:-2]])
            else:
                self.name += '.'+''.join([str(vpc[x]) for x in list(vpc)[:-1]])
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
        else:
            self.name+=f'_{1:.3f}'
        
        self.drop_ids = drop_ids
        self.scale_ids_factors = scale_ids_factors

        self.distillation_config['distillation']['parameters']['epochs'] = self.config['train']['epoches']
        self.dis_config_path = os.path.join(self.config['dis_config_dir'], self.name + '.toml')
        self.dis_model_path = os.path.join(self.config['dis_model_dir'], self.name +'.onnx')
        self.dis_log_path = os.path.join(self.config['dis_log_dir'], self.name + '.out')
        self.dis_slurm_path = os.path.join(self.config['dis_slurm_dir'], self.name + '.slurm')


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


    # verify network
    def verify(self, parameters, logger):
        configs = self.config
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
                verifier_parameters = '--bab.smart_branching True'
            else:
                v_name = v
                verifier_parameters = ""

            prop_dir = f'{configs["props_dir"]}/{self.name}/'

            if configs['name'] == 'acas':
                props = sorted(os.listdir(prop_dir))
                for p in props:
                    cmd = f'python -W ignore ./lib/DNNV/tools/resmonitor.py -T {time_limit} -M {memory_limit}'
                    cmd += f' ./scripts/run_dnnv.sh {prop_dir}/{p} --network N {self.dis_model_path} --{v_name} {verifier_parameters}'

                    count += 1
                    logger.info(f'Verifying with {v} {count}/{len(verifiers)}')
                    vpc = ''.join([str(self.vpc[x]) for x in self.vpc])

                    veri_log = f'{self.config["veri_log_dir"]}/{vpc}_{v}.out'
                    tmp_dir = f'"./tmp/{configs["name"]}_{configs["seed"]}_{vpc}_{v}"'

                    print(f'{self.config["veri_log_dir"]}/{vpc}_{v}_{p[0]}.out')
                    cmds = [cmd]
                    task = Task(cmds,
                                self.config['verify']['dispatch'],
                                "GDVB_Verify",
                                f'{self.config["veri_log_dir"]}/{vpc}_{v}_{p[0]}.out',
                                os.path.join(self.config['veri_slurm_dir'],f'{vpc}_{v}.slurm')
                    )
                    task.run()

            else:
                eps = (parameters['eps']*configs['verify']['eps'])[self.vpc['eps']]
                prop_levels = sorted([x for x in os.listdir(prop_dir) if '.py' in x])
                prop_levels = [ x for x in prop_levels if str(eps) in '.'.join(x.split('.')[2:-1])]
                prop = prop_levels[self.vpc['prop']]
                
                cmd = f'python -W ignore ./lib/DNNV/tools/resmonitor.py -T {time_limit} -M {memory_limit}'
                cmd += f' ./scripts/run_dnnv.sh {prop_dir}/{prop} --network N {self.dis_model_path} --{v_name} {verifier_parameters}'

                count += 1
                logger.info(f'Verifying with {v} {count}/{len(verifiers)}')
                vpc = ''.join([str(self.vpc[x]) for x in self.vpc])

                veri_log = f'{self.config["veri_log_dir"]}/{vpc}_{v}.out'
                tmp_dir = f'"./tmp/{configs["name"]}_{configs["seed"]}_{vpc}_{v}"'

                grb_license_file = self.config['verify']['GRB_LICENSE_FILE']

                os.environ["GRB_LICENSE_FILE"] = f"{grb_license_file}"
                cmds = [cmd]

                '''
                cmds = [f'export GRB_LICENSE_FILE="{grb_license_file}"',
                        f'export TMPDIR={tmp_dir}',
                        f'echo $TMPDIR',
                        f'mkdir -p $TMPDIR',
                        cmd,
                        f'rm -rf $TMPDIR']
                '''

                task = Task(cmds,
                            self.config['verify']['dispatch'],
                            "GDVB_Verify",
                            f'{self.config["veri_log_dir"]}/{vpc}_{v}.out',
                            os.path.join(self.config['veri_slurm_dir'],f'{vpc}_{v}.slurm')
                )
                task.run()


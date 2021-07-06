import os
import pathlib
import numpy as np
import toml
import re

from fractions import Fraction as F

from ..artifacts.ACAS import ACAS
from ..artifacts.MNIST import MNIST
from ..artifacts.CIFAR10 import CIFAR10
from ..artifacts.DAVE2 import DAVE2
from ..nn.layers import Dense, Conv, Transpose, Flatten
from ..dispatcher import Task
from ..pipeline.R4V import R4V
from ..pipeline.DNNV import DNNV
from ..pipeline.DNNF import DNNF


class VerificationProblem:
    def __init__(self, settings, vpc, verification_benchmark):
        self.settings = settings
        self.vpc = vpc
        self.verification_benchmark = verification_benchmark

        self.vp_name = ''
        self.net_name = ''
        for factor, level in [(x, vpc[x]) for x in vpc]:
            if factor in ['prop']:
                token = f'{factor}={level}_'
            else:
                token = f'{factor}={level:{self.settings.precision}}_'
            self.vp_name += token
            if factor not in ['eps', 'prop']:
                self.net_name += token
        self.vp_name = self.vp_name[:-1]
        self.net_name = self.net_name[:-1]

        self.prop_dir = os.path.join(self.settings.props_dir, self.net_name)

        source_dis_config = open(
            self.settings.dnn_configs['r4v_config'], 'r').read()
        self.distillation_config = toml.loads(source_dis_config)

        self.scale_input = False
        self.training_lost = {}
        self.verification_results = {}

    def set_distillation_strategies(self, dis_strats):
        self.settings.logger.debug('Distillation Strategies:')
        for i, x in enumerate(dis_strats):
            self.settings.logger.debug(f'{i}: {x}')

        drop_ids = []
        added_layers = []
        scale_ids_factors = []
        for ds in dis_strats:
            if ds[0] == 'drop':
                drop_ids += [ds[1]]
            elif ds[0] == 'add':
                added_layers += [ds[1]]
            elif ds[0] == 'scale':
                scale_ids_factors += [[ds[1], ds[2]]]
            elif ds[0] == 'scale_input':
                self.scale_input = True
                self.scale_input_factor = ds[1]
            else:
                assert False, 'Unknown strategy' + ds

        strategies = [x[0] for x in dis_strats]
        if 'scale' in strategies:
            self.net_name += f'_SF={dis_strats[strategies.index("scale")][2]:{self.settings.precision}}'
            self.vp_name += f'_SF={dis_strats[strategies.index("scale")][2]:{self.settings.precision}}'
        else:
            self.net_name += f'_{1:{self.settings.precision}}'
            self.vp_name += f'_{1:{self.settings.precision}}'

        self.drop_ids = drop_ids
        self.scale_ids_factors = scale_ids_factors
        self.added_layers = added_layers

        self.dis_config_path = os.path.join(
            self.settings.dis_config_dir, self.net_name + '.toml')
        self.dis_model_path = os.path.join(
            self.settings.dis_model_dir, self.net_name + '.onnx')
        self.dis_log_path = os.path.join(
            self.settings.dis_log_dir, self.net_name + '.out')
        if self.settings.training_configs['dispatch']['platform'] == 'slurm':
            self.dis_slurm_path = os.path.join(
                self.settings.dis_slurm_dir, self.net_name + '.slurm')
        else:
            self.dis_slurm_path = None

    # calculate neurons/layers
    def calc_order(self, order_by, orig_layers, input_shape):
        if order_by == 'nb_neurons':
            self.layers = []
            self.nb_neurons = []
            self.remaining_layer_ids = []
            self.fc_and_conv_kernel_sizes = []

            # create network
            for i in range(len(orig_layers)):
                if i in self.drop_ids:
                    self.layers += [None]
                    self.nb_neurons += [0]
                    self.fc_and_conv_kernel_sizes += [0]
                else:
                    tmp_layers = [x for x in self.layers if x is not None]
                    if not tmp_layers:
                        in_shape = input_shape
                    else:
                        in_shape = tmp_layers[-1].out_shape

                    ol = orig_layers[i]
                    if ol.type == 'FC':
                        size = ol.size
                        for x in self.scale_ids_factors:
                            if x[0] == i:
                                size = int(round(size * x[1]))
                                break
                        self.fc_and_conv_kernel_sizes += [size]
                        l = Dense(size, None, None, in_shape)
                        self.nb_neurons += [np.prod(l.out_shape)]
                    elif ol.type == 'Conv':
                        size = ol.size
                        for x in self.scale_ids_factors:
                            if x[0] == i:
                                size = int(round(size * x[1]))
                                break
                        self.fc_and_conv_kernel_sizes += [size]
                        l = Conv(size, None, None, ol.kernel_size,
                                 ol.stride, ol.padding, in_shape)
                        self.nb_neurons += [np.prod(l.out_shape)]
                    elif ol.type == 'Transpose':
                        l = Transpose(ol.order, in_shape)
                        self.fc_and_conv_kernel_sizes += [0]
                        self.nb_neurons += [0]
                    elif ol.type == 'Flatten':
                        l = Flatten(in_shape)
                        self.fc_and_conv_kernel_sizes += [0]
                        self.nb_neurons += [0]
                    else:
                        assert False
                    self.layers += [l]
                    self.remaining_layer_ids += [i]

            # add layers
            for layer in self.added_layers:
                if layer['layer_type'] == 'FullyConnected':
                    for layer_id in layer['layer_id']:
                        in_shape = self.layers[layer_id-1].out_shape
                        size = layer['parameters']
                        for x in self.scale_ids_factors:
                            if x[0] == i:
                                size = int(round(size * x[1]))
                                break
                        new_layer = Dense(size, None, None, in_shape)
                        self.layers.insert(layer_id, new_layer)
                        self.nb_neurons.insert(
                            layer_id, np.prod(new_layer.out_shape))
                        self.fc_and_conv_kernel_sizes.insert(layer_id, size)
                else:
                    raise NotImplementedError
            self.layers = [x for x in self.layers if x is not None]
        else:
            assert False

    # override comparators
    def __gt__(self, other):
        if self.order_by == 'nb_neurons' and other.order_by == 'nb_neurons':
            return np.sum(self.nb_neurons) > np.sum(other.nb_neurons)
        else:
            raise NotImplementedError()

    # writes the R4V training configs
    def write_training_configs(self):
        if 'epochs' in self.settings.training_configs:
            self.distillation_config['distillation']['parameters']['epochs'] = self.settings.training_configs['epochs']
        training_configs = toml.dumps(self.distillation_config)
        open(self.dis_config_path, 'w').write(training_configs)

        # write distillation strategies in order: scale_input > drop > add > scale_layer
        # TODO: use better toml api
        lines = ['']
        if self.scale_input:
            lines += ['[[distillation.strategies.scale_input]]']
            lines += [f'factor={self.scale_input_factor}\n']

        if self.drop_ids:
            lines += ['[[distillation.strategies.drop_layer]]']
            lines += ['layer_id=[' +
                      ', '.join([str(x) for x in self.drop_ids])+']']
            lines += ['']

        if self.added_layers:
            for layer in self.added_layers:
                lines += ['[[distillation.strategies.add_layer]]']
                lines += [f'layer_type="{layer["layer_type"]}"']
                lines += [f'parameters={layer["parameters"]}']
                lines += [f'activation_function="{layer["activation_function"]}"']
                lines += [f'layer_id={layer["layer_id"]}']
                lines += ['']

        if self.scale_ids_factors:
            lines += ['[[distillation.strategies.scale_layer]]']
            lines += ['layer_id=[' +
                      ', '.join([str(x[0]) for x in self.scale_ids_factors])+']']
            lines += ['factor=[{}]'.format(', '.join([str(x[1])
                                           for x in self.scale_ids_factors]))]
            lines += ['']

        lines += ['[distillation.student]']
        lines += ['path="'+self.dis_model_path+'"']
        lines = [x+'\n' for x in lines]

        open(self.dis_config_path, 'a').writelines(lines)

    # am I trained?
    def trained(self):
        trained = False
        if os.path.exists(self.dis_log_path):
            lines = open(self.dis_log_path, 'r').readlines()[-10:]
            for line in lines:
                if 'Process finished successfully' in line:
                    trained = True
                    break
        return trained

    # train network
    def train(self):
        if not self.settings.override and self.trained():
            self.settings.logger.info(f'Skipping trained network ...')
            return
        else:
            self.write_training_configs()

            cmd = R4V(["distill", "debug"]).execute([self.dis_config_path])
            cmds = [cmd]
            task = Task(cmds,
                        self.settings.training_configs['dispatch'],
                        "GDVB_Train",
                        self.dis_log_path,
                        self.dis_slurm_path
                        )
            self.settings.logger.debug(f'Command: {cmd}')
            print(cmd)
            task.run()

    def analyze_training(self):
        relative_loss = []
        if os.path.exists(self.dis_log_path):
            lines = open(self.dis_log_path).readlines()
            for line in lines:
                if 'validation error' in line:
                    relative_loss += [float(line.strip().split('=')[-1])]
        if len(relative_loss) != self.settings.training_configs['epochs']:
            self.settings.logger.warning(f"Training may not be finished. "
                                         f"({len(relative_loss)}/{self.settings.training_configs['epochs']})")
        return relative_loss

    def gen_prop(self):
        if isinstance(self.verification_benchmark.artifact, ACAS):
            prop_id = self.vpc['prop']
            self.verification_benchmark.artifact.generate_property(prop_id)

        elif isinstance(self.verification_benchmark.artifact, (MNIST, CIFAR10, DAVE2)):
            data_config = self.distillation_config['distillation']['data']
            prop_id = self.vpc['prop']

            if 'eps' in self.vpc:
                eps = F(self.vpc['eps']) * \
                    F(self.settings.verification_configs['eps'])
            else:
                eps = self.settings.verification_configs['eps']

            skip_layers = 0 if 'skip_layers' not in self.settings.verification_configs\
                else self.settings.verification_configs['skip_layers']

            pathlib.Path(self.prop_dir).mkdir(parents=True, exist_ok=True)

            self.verification_benchmark.artifact.generate_property(data_config,
                                                                   prop_id,
                                                                   eps,
                                                                   skip_layers,
                                                                   self.prop_dir,
                                                                   self.settings.seed)
        else:
            raise NotImplementedError

    # am I verified?
    def verified(self):
        log_path = self.veri_log_path[:-3] + 'err'
        verified = False
        if os.path.exists(log_path):
            lines = open(log_path).readlines()[-10:]
            for line in lines:
                if any(x in line for x in ['Timeout (terminating process)',
                                           'Process finished successfully',
                                           'Out of Memory (terminating process)']):
                    verified = True
                    break
        return verified

    # verify network
    def verify(self, tool, options):
        options = [options]
        if self.settings.verification_configs['debug']:
            options += ['debug']
        verifier = globals()[tool](options)

        time_limit = self.settings.verification_configs['time']
        memory_limit = self.settings.verification_configs['memory']

        net_path = self.dis_model_path
        self.veri_log_path = os.path.join(
            self.settings.veri_log_dir,
            f'{self.vp_name}_T={time_limit}_M={memory_limit}:{verifier.verifier_name}.out')

        if self.settings.verification_configs['dispatch']['platform'] == 'slurm':
            slurm_script_path = os.path.join(
                self.settings.veri_slurm_dir,
                f'{self.vp_name}_T={time_limit}_M={memory_limit}:{verifier.verifier_name}.slurm')
        else:
            slurm_script_path = None

        if not self.settings.override and self.verified():
            self.settings.logger.info('Skipping verified problem ...')
            return

        if 'eps' in self.vpc:
            eps = F(self.vpc['eps']) * \
                F(self.settings.verification_configs['eps'])
        else:
            eps = self.settings.verification_configs['eps']

        property_path = os.path.join(
            self.prop_dir, f"robustness_{self.vpc['prop']}_{eps}.py")

        cmd = f'python -W ignore ./lib/DNNV/tools/resmonitor.py -T {time_limit} -M {memory_limit} '
        cmd += verifier.execute([property_path, '--network N', net_path])
        cmds = [cmd]
        task = Task(cmds,
                    self.settings.verification_configs['dispatch'],
                    "GDVB_Verify",
                    self.veri_log_path,
                    slurm_script_path
                    )
        self.settings.logger.debug(f'Command: {cmd}')
        task.run()

    def analyze_verification(self):
        verification_results = {}
        verifiers = []
        for tool in self.settings.verification_configs['verifiers']:
            for options in self.settings.verification_configs['verifiers'][tool]:
                verifier = globals()[tool]([options])
                verifiers += [verifier]

        time_limit = self.settings.verification_configs['time']
        memory_limit = self.settings.verification_configs['memory']
        for verifier in verifiers:
            log_path = os.path.join(
                self.settings.veri_log_dir,
                f'{self.vp_name}_T={time_limit}_M={memory_limit}:{verifier.verifier_name}.out')

            if not os.path.exists(log_path):
                verification_answer = 'unrun'
                self.settings.logger.warning(f'Unrun: {log_path}')
                verification_time = -1
            else:
                LINES_TO_CHECK = 300
                lines_err = list(reversed(open(log_path, 'r').readlines()))
                lines_out = list(
                    reversed(open(os.path.splitext(log_path)[0] + '.err', 'r').readlines()))
                # lines = lines_out[:LINES_TO_CHECK] + lines_err[:LINES_TO_CHECK]
                lines = lines_out + lines_err

                verification_answer = None
                verification_time = None
                for i, l in enumerate(lines):

                    if re.match(r'INFO*', l):
                        continue

                    if re.search('Timeout', l):
                        verification_answer = 'timeout'
                        verification_time = time_limit
                        break

                    if re.search('Out of Memory', l):
                        verification_answer = 'memout'
                        for l in lines:
                            if re.search('Duration', l):
                                verification_time = float(l.split(' ')[9][:-2])
                                break
                        break

                    # if re.search('RuntimeError: view size is not compatible', l):
                    #    verification_answer = 'error'
                    #    verification_time = time_limit
                    #    break

                    if re.search(' result: ', l):
                        error_patterns = ['PlanetError',
                                          'ReluplexError',
                                          'ReluplexTranslatorError',
                                          'ERANError',
                                          'MIPVerifyTranslatorError',
                                          'NeurifyError',
                                          'NeurifyTranslatorError',
                                          'NnenumError',
                                          'NnenumTranslatorError',
                                          'MarabouError',
                                          'VerinetError',
                                          'MIPVerifyError']
                        if any(re.search(x, l) for x in error_patterns):
                            verification_answer = 'error'
                        # elif re.search('Return code: -11', l):
                        #    verification_answer = 'memout'
                        else:
                            verification_answer = l.strip().split(' ')[-1]
                            verification_time = float(
                                lines[i - 1].strip().split(' ')[-1])
                        break

                    # exceptions that DNNV didn't catch
                    # exception_patterns = ["Aborted         "]
                    # if any(re.search(x, l) for x in exception_patterns):
                    #    verification_answer = 'exception'
                    #    for l in rlines:
                    #        if re.search('Duration', l):
                    #            verification_time = float(l.split(' ')[9][:-2])
                    #            break
                    #    break

                    # failed jobs that are likely caused by server error
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
                                      '--- Logging error ---',
                                      'corrupted size vs. prev_size']
                    if any(re.search(x, l) for x in rerun_patterns):
                        verification_answer = 'rerun'
                        verification_time = -1
                        self.settings.logger.warning(
                            f'Failed job({verification_answer}): {log_path}')
                        break

            if not verification_answer and (i + 1 in [LINES_TO_CHECK, len(lines)] or len(lines) == 0):
                verification_answer = 'undetermined'
                verification_time = -1
                self.settings.logger.warning(f'Undetermined job: {log_path}')

            assert verification_answer, verification_time
            assert verification_answer in ['sat', 'unsat', 'unknown', 'error', 'timeout',
                                           'memout', 'exception', 'rerun', 'unrun', 'undetermined'],\
                f'{verification_answer}:{log_path}'

            verification_results[verifier.verifier_name] = [
                verification_answer, verification_time]

        self.verification_results = verification_results
        return verification_results

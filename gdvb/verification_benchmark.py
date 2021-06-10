import os
import random
import pickle
import copy
import numpy as np
from tqdm import tqdm

from .artifacts.ACAS import ACAS
from .artifacts.MNIST import MNIST
from .artifacts.CIFAR10 import CIFAR10
from .artifacts.DAVE2 import DAVE2
from .nn.layers import Dense, Conv

from decimal import Decimal as D
from .verification_problem import VerificationProblem


class VerificationBenchmark:

    def __init__(self, name, dnn_configs, ca_configs, settings):
        self.name = name
        self.settings = settings
        self.artifact = self._create_artifact(dnn_configs)
        self.settings.logger.info('Computing Factors')
        (self.parameters, self.fc_ids, self.conv_ids) = self._gen_parameters(ca_configs)
        self._debug_layer()
        self.settings.logger.info('Computing Covering Array')
        self.ca = self._gen_ca(ca_configs)
        self.settings.logger.info('Computing DNN Specifications')
        self.verification_problems = self._gen_network_specifications()

        self.training_loss = {}
        self.verification_results = {}

    def _create_artifact(self, dnn_configs):
        if dnn_configs['artifact'] in globals():
            artifact = globals()[dnn_configs['artifact']](dnn_configs)
        else:
            raise NotImplementedError(f'Unimplemented artifact: {dnn_configs["artifact"]}')
        return artifact

    def _gen_parameters(self, ca_configs):
        # calculate fc/conv ids
        fc_ids = []
        conv_ids = []
        for i, x in enumerate(self.artifact.layers):
            if isinstance(x, Conv):
                conv_ids += [i]
            elif isinstance(x, Dense):
                fc_ids += [i]
        fc_ids = fc_ids[:-1]  # remove output layer
        parameters = {}
        # parse the parameters in the expressive way
        if 'explicit' in ca_configs['parameters']:
            for x in ca_configs['parameters']['explicit']:
                parameters[x] = ca_configs['parameters']['explicit'][x]
        # parse the parameters described in number of levels and ranges
        else:
            for key in ca_configs['parameters']['level']:
                level_min = ca_configs['parameters']['range'][key][0]
                level_max = ca_configs['parameters']['range'][key][1]
                level_size = ca_configs['parameters']['level'][key]
                if level_size == 1:
                    assert level_min == level_max
                    level = np.array([level_min])
                else:
                    level_range = D(str(level_max)) - D(str(level_min))
                    level_step = D(str(level_range)) / D(str(level_size - 1))
                    level = np.arange(D(str(level_min)), D(str(level_max)) + level_step, D(str(level_step)))
                    level = np.array(level, dtype=np.float)
                    if key == 'prop':
                        level = np.array(level, dtype=np.int)

                    assert len(level) == level_size

                parameters[key] = level
                # make sure all parameters are passed
                assert len(parameters[key]) == ca_configs['parameters']['level'][key]
        return parameters, fc_ids, conv_ids

    def _debug_layer(self):
        # debug remaining layers
        if 'fc' in self.parameters:
            prm_str = "Possible remaining # of FC layers: "
            rml = sorted([str(int(round(len(self.fc_ids) * (self.parameters['fc'][i]))))
                          for i in range(len(self.parameters['fc']))])
            prm_str += ' '.join(rml)
            self.settings.logger.debug(prm_str)

        if 'conv' in self.parameters:
            prm_str = "Possible remaining # of Conv layers: "
            rml = sorted([str(int(round(len(self.conv_ids) * (self.parameters['conv'][i]))))
                          for i in range(len(self.parameters['conv']))])
            prm_str += ' '.join(rml)
            self.settings.logger.debug(prm_str)

        # print factor and levels
        self.settings.logger.debug('Factor and levels:')
        for key in self.parameters:
            self.settings.logger.debug(f'{key}: {self.parameters[key]}')

    def _gen_ca(self, ca_configs):
        # compute the covering array
        lines = [
            '[System]',
            f'Name: {self.name}',
            '',
            '[Parameter]']

        for key in self.parameters:
            bits = [str(x) for x in range(len(self.parameters[key]))]
            bits = ','.join(bits)
            lines += [f'{key}(int) : {bits}']

        if 'constraints' in ca_configs:
            lines += ['[Constraint]']
            constraints = ca_configs['constraints']['value']
            for con in constraints:
                lines += [con]

        lines = [x + '\n' for x in lines]

        strength = ca_configs['strength']

        ca_config_path = os.path.join(self.settings.root, 'ca_config.txt')
        ca_path = os.path.join(self.settings.root, 'ca.txt')

        open(ca_config_path, 'w').writelines(lines)

        acts_path = './lib/acts.jar'
        if not os.path.exists(acts_path):
            raise FileNotFoundError(f'CA generator ACTS is not found at :{acts_path}')
        cmd = f'java  -Ddoi={strength} -jar {acts_path} {ca_config_path} {ca_path} > /dev/null'
        os.system(cmd)

        lines = open(ca_path, 'r').readlines()

        vp_configs = []
        i = 0
        while i < len(lines):
            l = lines[i]
            if 'Number of configurations' in l:
                nb_tests = int(l.strip().split(' ')[-1])
                self.settings.logger.info(f'# Tests: {nb_tests}')

            if 'Configuration #' in l:
                vp = []
                for j in range(len(self.parameters)):
                    l = lines[j + i + 2]
                    vp += [int(l.strip().split('=')[-1])]
                assert len(vp) == len(self.parameters)
                vp_configs += [vp]
                i += j + 2
            i += 1
        assert len(vp_configs) == nb_tests
        vp_configs = sorted(vp_configs)

        vp_configs_ = []
        for vpc in vp_configs:
            assert len(vpc) == len(self.parameters)
            tmp = {}
            for i, key in enumerate(self.parameters):
                tmp[key] = self.parameters[key][vpc[i]]
            vp_configs_ += [tmp]
        vp_configs = vp_configs_

        return vp_configs

    def _gen_network_specifications(self):
        network_specifications = []
        for vpc in self.ca:
            self.settings.logger.debug(f'Configuring verification problem: {vpc}')
            self.settings.logger.debug('----------Original Network----------')
            self.settings.logger.debug(f'Number neurons: {np.sum(self.artifact.onnx.nb_neurons)}')
            for i, x in enumerate(self.artifact.layers):
                self.settings.logger.debug(f'{i}: {x}')

            # factor: neu
            if 'neu' in vpc:
                neuron_scale_factor = vpc['neu']
            else:
                neuron_scale_factor = 1

            # factor: fc, conv
            drop_ids = []
            add_ids = []
            layers_add = []
            if 'fc' in vpc:
                if vpc['fc'] < 1:
                    # randomly select layers to drop
                    if self.settings.training_configs['drop_scheme'] == 'random':
                        drop_fc_ids = sorted(random.sample(self.fc_ids, int(
                            round(len(self.fc_ids) * (1 - vpc['fc'])))))
                        drop_ids += drop_fc_ids
                    else:
                        raise NotImplementedError(f'Unsupported drop scheme: '
                                                  f'{self.settings.training_configs["drop_scheme"]}')
                elif vpc['fc'] > 1:
                    # append fully connected layers to the last hidden layer
                    # nb_neurons = nb_neurons in the last hidden layer
                    if self.settings.training_configs['add_scheme'] == 'same_as_last_relu':
                        last_layer_id = np.max(self.fc_ids)
                        nb_layer_to_add = np.arange(1, round(len(self.fc_ids) * (vpc['fc'] - 1))+1, 1)
                        fc_add_ids = [x + last_layer_id for x in nb_layer_to_add]
                        nb_neurons = self.artifact.layers[last_layer_id].size
                        layer = {'layer_type': 'FullyConnected',
                                 'parameters': nb_neurons,
                                 'activation_function': 'relu',
                                 'layer_id': fc_add_ids}
                        layers_add += [layer]
                        add_ids += fc_add_ids
                    else:
                        raise NotImplementedError(f'Unsupported drop scheme: '
                                                  f'{self.settings.training_configs["drop_scheme"]}')
                else:
                    pass

            if 'conv' in vpc:
                if vpc['conv'] < 1:
                    # randomly select layers to drop
                    if self.settings.training_configs['drop_scheme'] == 'random':
                        drop_conv_ids = sorted(random.sample(self.conv_ids, int(
                            round(len(self.conv_ids) * (1 - vpc['conv'])))))
                        drop_ids += drop_conv_ids

                    else:
                        raise NotImplementedError(f'Unsupported add scheme: '
                                                  f'{self.settings.training_configs["add_scheme"]}')
                elif vpc['conv'] > 1:
                    # append convolutional layers to the last hidden layer
                    # nb_neurons = nb_neurons in the last hidden layer
                    if self.settings.training_configs['add_scheme'] == 'same_as_last_relu':
                        last_layer_id = np.max(self.conv_ids)
                        nb_layer_to_add = np.arange(1, round(len(self.conv_ids) * (vpc['conv'] - 1))+1, 1)
                        conv_add_ids = [x + last_layer_id for x in nb_layer_to_add]
                        last_layer = self.artifact.layers[last_layer_id]
                        assert isinstance(last_layer, Conv)
                        nb_kernels = last_layer.size
                        kernel_size = last_layer.kernel_size
                        stride = last_layer.stride
                        padding = last_layer.padding
                        layer = {'layer_type': 'FullyConnected',
                                 'parameters': [nb_kernels, kernel_size, stride, padding],
                                 'activation_function': 'relu',
                                 'layer_id': conv_add_ids}
                        layers_add += [layer]
                        add_ids += conv_add_ids
                    else:
                        raise NotImplementedError(f'Unsupported add scheme: '
                                                  f'{self.settings.training_configs["add_scheme"]}')
                else:
                    pass

            n = VerificationProblem(self.settings, vpc, self)
            dis_strats = [['drop', x] for x in drop_ids]
            dis_strats += [['add', x] for x in layers_add]

            # calculate data transformations, input dimensions
            transform = n.distillation_config['distillation']['data']['transform']['student']
            height = transform['height']
            width = transform['width']
            assert height == width

            # input dimension
            if 'idm' in vpc:
                id_f = vpc['idm']
                new_height = int(round(np.sqrt(height * width * id_f)))
                transform['height'] = new_height
                transform['width'] = new_height
                if new_height != height:
                    dis_strats += [['scale_input', new_height / height]]
            else:
                new_height = height

            # input domain size
            if 'ids' in vpc:
                mean = transform['mean']
                max_value = transform['max_value']
                min_value = transform['min_value']
                ids_f = vpc['ids']

                transform['mean'] = [float(x * ids_f) for x in mean]
                transform['max_value'] = float(max_value * ids_f)
                transform['min_value'] = float(min_value * ids_f)

            if self.artifact.onnx.input_format == 'NCHW':
                nb_channel = self.artifact.onnx.input_shape[0]
            elif self.artifact.onnx.input_format == 'NHWC':
                nb_channel = self.artifact.onnx.input_shape[2]
            else:
                raise ValueError(f'Unrecognized ONNX input format: {self.artifact.onnx.input_format}')

            # set up new network with added and dropped layers
            n.set_distillation_strategies(dis_strats)
            input_shape = [nb_channel, new_height, new_height]
            n.calc_order('nb_neurons', self.artifact.layers, input_shape)

            # calculate real scale factors
            if 'neu' in vpc:
                neuron_scale_factor = (np.sum(self.artifact.onnx.nb_neurons[:-1]) * neuron_scale_factor) / np.sum(n.nb_neurons[:-1])
            elif 'fc' in vpc or 'conv' in vpc:
                neuron_scale_factor = np.sum(self.artifact.onnx.nb_neurons[:-1]) / np.sum(n.nb_neurons[:-1])

            # assign scale factors
            if neuron_scale_factor != 1:
                # calculate scale ids
                if 'neu' in vpc or 'fc' in vpc or 'conv' in vpc:
                    scale_ids = set(self.fc_ids + self.conv_ids)
                    scale_ids = set(scale_ids) - set(drop_ids)
                    scale_ids = list(scale_ids)
                    # print(scale_ids)

                    if add_ids:
                        for add_id in add_ids:
                            for i in range(len(scale_ids)):
                                if scale_ids[i] >= add_id:
                                    scale_ids[i] += 1
                            scale_ids = sorted(scale_ids + [add_id])
                            # print(scale_ids)
                    # print(scale_ids)
                    # print()
                else:
                    scale_ids = []
                self.settings.logger.debug('Computing layer scale factors ...')
                self.settings.logger.debug(f'Layers to Add: {add_ids}, Delete: {drop_ids}.')
                self.settings.logger.debug(f'Layers to Scale: {scale_ids}.')
                for x in scale_ids:
                    assert n.fc_and_conv_kernel_sizes[x] > 0
                    if int(n.fc_and_conv_kernel_sizes[x] * neuron_scale_factor) == 0:
                        self.settings.logger.warn('Detected small layer scale factor, layer size is rounded up to 1.')
                        dis_strats += [['scale', x, 1 / n.fc_and_conv_kernel_sizes[x]]]
                    else:
                        dis_strats += [['scale', x, neuron_scale_factor]]
                n = VerificationProblem(self.settings, vpc, self)
                n.set_distillation_strategies(dis_strats)
                n.calc_order('nb_neurons', self.artifact.layers, input_shape)

            n.distillation_config['distillation']['data']['transform']['student'] = transform
            network_specifications += [n]
            self.settings.logger.debug('----------New Network----------')
            self.settings.logger.debug(f'Number neurons: {np.sum(n.nb_neurons)}')
            for i, x in enumerate(n.layers):
                self.settings.logger.debug(f'{i}: {x}')

        self.settings.logger.info(f'# NN: {len(network_specifications)}')
        return network_specifications

    def train(self):
        self.settings.logger.info('Training ...')
        # filter repeated network specifications
        nets_to_train = {x.net_name: x for x in self.verification_problems}
        nets_to_train = [nets_to_train[x] for x in nets_to_train]

        for i in tqdm(range(len(nets_to_train)), desc="Training ... ", ascii=False):
            n = nets_to_train[i]
            self.settings.logger.info(f'Training network: {n.net_name} ...')
            n.train()

    def gen_props(self):
        self.settings.logger.info('Generating properties ...')
        for i in tqdm(range(len(self.verification_problems)), desc="Generating ... ", ascii=False):
            vp = self.verification_problems[i]
            vp.gen_prop()

    def verify(self):
        self.settings.logger.info('Verifying ...')
        vp_tool_verifiers = []
        for vp in self.verification_problems:
            for tool in self.settings.verification_configs['verifiers']:
                for options in self.settings.verification_configs['verifiers'][tool]:
                    vp_tool_verifiers += [(vp, tool, options)]

        for i in tqdm(range(len(vp_tool_verifiers)), desc="Verifying ... ", ascii=False):
            vp = vp_tool_verifiers[i][0]
            tool = vp_tool_verifiers[i][1]
            options = vp_tool_verifiers[i][2]

            self.settings.logger.info(f'Verifying {vp.vp_name} with {tool}:[{options}] ...')
            vp.gen_prop()
            vp.verify(tool, options)

    def analyze(self):
        for vp in self.verification_problems:
            self.training_loss[vp.vp_name] = vp.analyze_training()

        for vp in self.verification_problems:
            self.verification_results[vp.vp_name] = vp.analyze_verification()

    def save_results(self):
        save_path_prefix = os.path.join(self.settings.root, f'{self.name}_{self.settings.seed}')

        train_loss_lines = []
        # save training losses as csv
        for x in self.training_loss:
            line = f'{x},'
            for xx in self.training_loss[x]:
                line = line + f'{xx}, '
            train_loss_lines += [f'{line}\n']

        with open(f'{save_path_prefix}_train_loss.csv', 'w') as handle:
            handle.writelines(train_loss_lines)

        # transpose verification results
        # from: results[problems][verifiers]
        # to:   results[verifiers][problems]
        results = {}
        verifiers = [x for x in list(self.verification_results.values())[0]]
        for p in self.verification_results:
            for v in verifiers:
                if v not in results.keys():
                    results[v] = {}
                assert p not in results[v].keys()
                results[v][p] = self.verification_results[p][v]

        # save verification results as pickle
        with open(f'{save_path_prefix}_verification_results.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # calculate scr and par2 scores

        scr_dict = {}
        par_2_dict = {}
        for v in verifiers:
            sums = []
            scr = 0
            for p in results[v]:
                verification_answer = results[v][p][0]
                verification_time = results[v][p][1]
                if verification_answer in ["sat", "unsat"]:
                    scr += 1
                    sums += [verification_time]
                elif verification_answer in ['unknown', 'error', 'timeout', 'memout', 'exception',
                                             'rerun', 'torun', 'unrun', 'undetermined']:
                    sums += [self.settings.verification_configs['time'] * 2]
                else:
                    assert False
            par_2 = np.mean(np.array(sums))

            if v not in scr_dict.keys():
                scr_dict[v] = [scr]
                par_2_dict[v] = [par_2]
            else:
                scr_dict[v] += [scr]
                par_2_dict[v] += [par_2]

        print('')
        print('|{:>15} | {:>15} | {:>15}|'.format('Verifier', 'SCR', 'PAR-2'))
        print('|----------------|-----------------|----------------|')
        for v in verifiers:
            print('|{:>15} | {:>15} | {:>15.2f}|'.format(v, scr_dict[v][0], round(float(par_2_dict[v][0]), 2)))

import copy
import numpy as np

from fractions import Fraction as F

from ..verification_benchmark import VerificationBenchmark

from .step import EvoStep


class EvoBench:
    def __init__(self, seed_benchmark):
        self.seed_benchmark = seed_benchmark
        self.logger = seed_benchmark.settings.logger
        self._init_parameters()
    
    def _init_parameters(self):
        self.steps = []
        self.global_min = {}
        self.global_max = {}

        for p in self.seed_benchmark.settings.evolutionary['parameters']:
            self.global_min[p] = None
            self.global_max[p] = None
        

    def run(self):
        verification_benchmark = self.seed_benchmark

        benchmark_name = verification_benchmark.settings.name
        dnn_configs = verification_benchmark.settings.dnn_configs
        evo_configs = verification_benchmark.settings.evolutionary
        arity = evo_configs['arity']
        inflation_rate = evo_configs['inflation_rate']
        deflation_rate = evo_configs['deflation_rate']

        self.steps = [EvoStep(self.seed_benchmark)]

        for i in range(evo_configs['iterations']):
            evo_step = self.steps[-1]
            neus = sorted(set([x['neu'] for x in evo_step.benchmark.ca]))
            fcs = sorted(set([x['fc'] for x in evo_step.benchmark.ca]))
            print(i, neus, fcs, '\n')

            evo_step.proceed()

            next_ca_configs = self.evolve(evo_step, evo_configs['parameters'], arity, inflation_rate, deflation_rate, i)

            next_evo_step = EvoStep(VerificationBenchmark(f'{benchmark_name}_{i}',
                                                dnn_configs,
                                                next_ca_configs,
                                                evo_step.benchmark.settings))

            self.steps += [next_evo_step]


    def evolve(self, evo_step, parameters_to_change, arity, inflation_rate, deflation_rate, iteration):
        evo_step.evaluate(parameters_to_change)
        evo_step.plot(parameters_to_change, iteration)

        ca_configs = evo_step.benchmark.ca_configs
        ca_configs_new = copy.deepcopy(ca_configs)


        for i, param in enumerate(parameters_to_change):
            print(f'Working on factor: {param}')

            axes = list(range(len(parameters_to_change)))
            axes.remove(i)
            axes = [i] + axes
            raw = list(evo_step.nb_solved.values())[0].transpose(axes)

            nb_property = ca_configs['parameters']['level']['prop']
            min_cut = [np.all(x == nb_property) for x in raw]
            min_cut = min_cut.index(False) if False in min_cut else None
            max_cut = [np.all(x == 0) for x in reversed(raw)]
            max_cut = max_cut.index(False) if False in max_cut else None

            nb_levels = ca_configs['parameters']['level'][param]
            level_min = F(ca_configs['parameters']['range'][param][0])
            level_max = F(ca_configs['parameters']['range'][param][1])

            if min_cut is not None and min_cut > 0:
                g_min = np.arange(level_min, level_max+level_min, (level_max-level_min)/(nb_levels-1))[min_cut-1]
                if self.global_min[param] is None:
                    self.global_min[param] = g_min
                else:
                    if g_min > self.global_min[param]:
                        self.global_min[param] = g_min
            if max_cut is not None and max_cut > 0:
                g_max = np.arange(level_min, level_max+level_min, (level_max-level_min)/(nb_levels-1))[(nb_levels-1)-(max_cut-1)]
                if self.global_max[param] is None:
                    self.global_max[param] = g_max
                else:
                    if g_max < self.global_max[param]:
                        self.global_max[param] = g_max

            print(f'old CA configs: L({nb_levels}) [({level_min}),({level_max})]')
            print(raw)
            print('min-max:', min_cut, max_cut)

            # all too easy: inflate
            # all too hard: deflate
            if min_cut == None or max_cut == None:
                if min_cut == None:
                    assert max_cut != None
                    rate = inflation_rate
                elif max_cut == None:
                    assert min_cut != None
                    rate = deflation_rate
                else:
                    assert False

                nb_levels = nb_levels * (arity - 1)
                level_min = level_min * F(rate)
                level_max = level_max * F(rate)
                if self.global_min[param] and level_min < self.global_min[param]:
                    level_min = self.global_min[param]
                if self.global_max[param] and level_max > self.global_max[param]:
                    level_max = self.global_max[param]

                new_step = (level_max - level_min) / (nb_levels - 1)
                if param == 'fc':
                    min_step = 1 / F(len(evo_step.benchmark.fc_ids))
                elif param == 'conv':
                    min_step = 1 / F(len(evo_step.benchmark.conv_ids))
                else:
                    min_step = 0
                if new_step < min_step:
                    self.logger.warn(f'new_step:{new_step} < min_step:{min_step}')
                    nb_levels = int((level_max - level_min) / min_step) + 1

            else:
                old_step = (level_max-level_min) / (nb_levels - 1)
                new_step = old_step / arity

                if param == 'fc':
                    min_step = 1/F(len(evo_step.benchmark.fc_ids))
                elif param == 'conv':
                    min_step = 1/F(len(evo_step.benchmark.conv_ids))
                else:
                    min_step = 0
                if new_step < min_step:
                    self.logger.warn(f'new_step:{new_step} < min_step:{min_step}')
                    new_step = min_step

                print(f'old step: {old_step}, new step: {new_step}')

                level_min = level_min + min_cut * old_step - new_step
                if level_min == 0:
                    self.logger.warn(f'Removed "0" level.')
                    level_min += new_step
                level_max = level_max - max_cut * old_step + new_step

                if self.global_min[param] and level_min < self.global_min[param]:
                    level_min = self.global_min[param]
                if self.global_max[param] and level_max > self.global_max[param]:
                    level_max = self.global_max[param]

                nb_levels = int((level_max-level_min)/new_step + 1)

            print(f'New CA configs: L({nb_levels}) [({level_min}),({level_max})]')

            assert level_min > 0
            ranges = [level_min, level_max]
            ca_configs_new['parameters']['level'][param] = nb_levels
            ca_configs_new['parameters']['range'][param] = ranges

        return ca_configs_new



        exit()

        # 1) all too easy? inflate the benchmark
        # 2) all too hard? deflate the benchmark
        self.logger.debug(f'evolving ... with {evolve_strategy}')
        if evolve_strategy in ['deflate', 'inflate']:
            print('Deflate, inflate ...')
            if evolve_strategy == 'deflate':
                factor = deflation_rate
            elif evolve_strategy == 'inflate':
                factor = inflation_rate
            else:
                assert False

            for param in parameters_to_change:
                nb_levels = ca_configs['parameters']['level'][param] * (arity - 1)
                level_min = F(ca_configs['parameters']['range'][param][0]) * F(factor)
                level_max = F(ca_configs['parameters']['range'][param][1]) * F(factor)

                new_step = (level_max-level_min) / (nb_levels - 1)
                if param == 'fc':
                    min_step = 1/F(len(benchmark.fc_ids))
                elif param == 'conv':
                    min_step = 1/F(len(benchmark.conv_ids))
                else:
                    min_step = 0
                if new_step < min_step:
                    benchmark.settings.logger.debug(f'new_step < min_step')
                    nb_levels = int((level_max - level_min)/min_step) + 1

                ranges = [level_min, level_max]
                ca_configs_new['parameters']['level'][param] = nb_levels
                ca_configs_new['parameters']['range'][param] = ranges

        # 3) in between? zoom in the benchmark
        elif evolve_strategy == 'zoom in':
            print('Zoom in ...')
            for i, param in enumerate(parameters_to_change):
                nb_levels = ca_configs['parameters']['level'][param]
                level_min = F(ca_configs['parameters']['range'][param][0])
                level_max = F(ca_configs['parameters']['range'][param][1])

                print('Working on factor: ', param)
                print(f'old CA configs: L({nb_levels}) [({level_min}),({level_max})]')

                axes = list(range(len(parameters_to_change)))
                axes.remove(i)
                axes = [i] + axes
                raw = list(solved_per_verifiers.values())[0].transpose(axes)

                print(raw)

                if set([np.all(x == 0) for x in raw]) == {False}:
                    nb_levels = nb_levels * (arity - 1)
                    level_min = level_min * F(inflation_rate)
                    level_max = level_max * F(inflation_rate)

                    new_step = (level_max - level_min) / (nb_levels - 1)
                    if param == 'fc':
                        min_step = 1 / F(len(benchmark.fc_ids))
                    elif param == 'conv':
                        min_step = 1 / F(len(benchmark.conv_ids))
                    else:
                        min_step = 0
                    if new_step < min_step:
                        benchmark.settings.logger.debug(f'new_step < min_step')
                        nb_levels = int((level_max - level_min) / min_step) + 1
                else:
                    nb_property = ca_configs['parameters']['level']['prop']

                    for x, row in enumerate(raw):
                        if np.all(row == nb_property):
                            continue
                        else:
                            break
                    min_id = x
                    min_id2 = [np.all(x == nb_property) for x in raw].index(False)
                    assert min_id == min_id2

                    for x, row in enumerate(reversed(raw)):
                        if np.all(row == 0):
                            continue
                        else:
                            break
                    max_id = x
                    max_id2 = [np.all(x == 0) for x in reversed(raw)].index(False)
                    assert max_id == max_id2

                    assert min_id is not None
                    assert max_id is not None

                    print('min-max', min_id, max_id)

                    old_step = (level_max-level_min) / (nb_levels - 1)
                    new_step = old_step / arity
                    if param == 'fc':
                        min_step = 1/F(len(benchmark.fc_ids))
                    elif param == 'conv':
                        min_step = 1/F(len(benchmark.conv_ids))
                    else:
                        min_step = 0

                    if new_step < min_step:
                        new_step = min_step
                    # print(min_step)

                    # print(old_step, new_step)
                    print('old step: ', old_step, 'new step: ', new_step)

                    level_min = level_min + min_id * old_step - new_step
                    if level_min == 0:
                        level_min += new_step
                    level_max = level_max - max_id * old_step + new_step
                    nb_levels = int((level_max-level_min)/new_step + 1)

                print(f'New CA configs: L({nb_levels}) [({level_min}),({level_max})]')

                assert level_min > 0
                ranges = [level_min, level_max]
                ca_configs_new['parameters']['level'][param] = nb_levels
                ca_configs_new['parameters']['range'][param] = ranges
        else:
            assert False

        return ca_configs_new



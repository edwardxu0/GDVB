import copy
import numpy as np

from fractions import Fraction as F
from pathlib import Path

from ..core.verification_benchmark import VerificationBenchmark

from .evo_step import EvoStep

from ..plot.pie_scatter import PieScatter2D


class EvoBench:
    def __init__(self, seed_benchmark):
        self.logger = seed_benchmark.settings.logger
        self.seed_benchmark = seed_benchmark
        self.evo_configs = seed_benchmark.settings.evolutionary
        self.evo_params = self.evo_configs['parameters']
        assert len(self.evo_params) == 2
        self._init_parameters()

    def _init_parameters(self):
        self.steps = []
        self.global_min = {}
        self.global_max = {}
        self.res = {}

        for p in self.seed_benchmark.settings.evolutionary['parameters']:
            self.global_min[p] = False
            self.global_max[p] = False

    def run(self):
        verification_benchmark = self.seed_benchmark

        benchmark_name = verification_benchmark.settings.name
        dnn_configs = verification_benchmark.settings.dnn_configs

        self.steps += [EvoStep(self.seed_benchmark,
                               self.evo_configs['parameters'])]

        for i in range(self.evo_configs['iterations']):
            evo_step = self.steps[-1]

            # --
            neus = sorted(set([x['neu'] for x in evo_step.benchmark.ca]))
            fcs = sorted(set([x['fc'] for x in evo_step.benchmark.ca]))
            print(f"evo_iter={i}, neu={neus}, fc={fcs} \n")
            # --

            evo_step.forward()
            evo_step.evaluate()
            evo_step.plot(i+1)
            self.collect_res(evo_step)

            next_ca_configs, stop = self.evolve(evo_step)
            if stop:
                break

            next_benchmark = VerificationBenchmark(f'{benchmark_name}_{i}',
                                                   dnn_configs,
                                                   next_ca_configs,
                                                   evo_step.benchmark.settings)

            next_evo_step = EvoStep(
                next_benchmark, self.evo_configs['parameters'])

            self.steps += [next_evo_step]

        self.plot()
        self.logger.info('Evo bench finished successfuly!')

    def evolve(self, evo_step):

        ca_configs = evo_step.benchmark.ca_configs

        # a decision maker: determins what actions to take in the next step?
        actions = []
        for i, f in enumerate(evo_step.factors):
            f = copy.deepcopy(f)
            print(f'Working on factor: {f}')

            axes = list(range(len(self.evo_params)))
            axes.remove(i)
            axes = [i] + axes

            # TODO: only supports one([0]) verifier per time
            raw = list(evo_step.nb_solved.values())[0].transpose(axes)
            print(raw)

            nb_property = ca_configs['parameters']['level']['prop']
            min_cut = [np.all(x == nb_property) for x in raw]
            min_cut = min_cut.index(False) if False in min_cut else None
            max_cut = [np.all(x == 0) for x in reversed(raw)]
            max_cut = max_cut.index(False) if False in max_cut else None
            print(f'min cut:{min_cut}; max cut:{max_cut}.')

            if max_cut == 0:
                action = 'expand'
            else:
                action = 'stop'

            actions += [action]

        actions = np.array(actions)

        print(actions)
        if np.all(actions == 'stop'):
            next_ca_configs = copy.deepcopy(ca_configs)
            stop = True
        elif np.any(actions == 'expand'):
            next_ca_configs = self.expanding(evo_step, actions)

            _stop = []
            for p in self.evo_params:
                this_end = ca_configs['parameters']['range'][p][1]
                next_end = next_ca_configs['parameters']['range'][p][1]
                _stop += [this_end == next_end]
            _stop = np.array(_stop)
            stop = np.all(_stop == True)
            print(_stop, stop)
        else:
            assert False

        return next_ca_configs, stop

    def expanding(self, evo_step, actions):
        ca_configs = evo_step.benchmark.ca_configs
        ca_configs_new = copy.deepcopy(ca_configs)

        arity = self.evo_configs['arity']
        inflation_rate = self.evo_configs['inflation_rate']
        deflation_rate = self.evo_configs['deflation_rate']

        parameters_upper_bounds = self.evo_configs['parameters_upper_bounds']

        for i, f in enumerate(evo_step.factors):
            f = copy.deepcopy(f)
            if actions[i] == 'expand':
                start = f.start*inflation_rate
                end = f.end*inflation_rate

                if f.type in parameters_upper_bounds:
                    end = min(end, F(parameters_upper_bounds[f.type]))
                print(start, end)
                f.set_start_end(start, end)
            start, end, levels = f.get()
            ca_configs_new['parameters']['level'][f.type] = levels
            ca_configs_new['parameters']['range'][f.type] = [start, end]
        return ca_configs_new

    def sharpening(self):
        pass

    def evolve2(self, evo_step):
        evo_params = self.evo_configs['parameters']
        assert len(evo_params) == 2
        arity = self.evo_configs['arity']
        inflation_rate = self.evo_configs['inflation_rate']
        deflation_rate = self.evo_configs['deflation_rate']

        ca_configs = evo_step.benchmark.ca_configs
        ca_configs_new = copy.deepcopy(ca_configs)

        for i, f in enumerate(evo_step.factors):
            f = copy.deepcopy(f)
            print(f'Working on factor: {f}')

            axes = list(range(len(evo_params)))
            axes.remove(i)
            axes = [i] + axes
            raw = list(evo_step.nb_solved.values())[0].transpose(axes)
            print(raw)

            # TODO ???
            # cut levels that are too easy and too hard
            nb_property = ca_configs['parameters']['level']['prop']
            min_cut = [np.all(x == nb_property) for x in raw]
            min_cut = min_cut.index(False) if False in min_cut else None
            max_cut = [np.all(x == 0) for x in reversed(raw)]
            max_cut = max_cut.index(False) if False in max_cut else None
            print('min-max:', min_cut, max_cut)

            # TODO ???
            # set global_min and global_max
            if min_cut and min_cut > 0:
                self.global_min[f.type] = True
            if max_cut and max_cut > 0:
                self.global_max[f.type] = True

            # find lower and upper boders
            if not self.global_min[f.type] and not self.global_max[f.type]:
                subdivision = False
                f.set_start_end(f.start*deflation_rate, f.end*inflation_rate)
            elif not self.global_min[f.type] and self.global_max[f.type]:
                subdivision = False
                f.set_start(f.start*deflation_rate)
            elif self.global_min[f.type] and not self.global_max[f.type]:
                subdivision = False
                f.set_end(f.end*inflation_rate)
            else:
                subdivision = True

            # cut easy or hard
            new_start = f.start + min_cut * f.step if min_cut else None
            new_end = f.end - max_cut * f.step if max_cut else None
            if new_start and new_end:
                f.set_start_end(new_start, new_end)
            elif new_start and not new_end:
                f.set_start(new_start)
            elif not new_start and new_end:
                f.set_end(new_end)
            else:
                pass

            # subdivide the level plane
            if subdivision:
                f.subdivision(arity)

            start, end, levels = f.get()
            ca_configs_new['parameters']['level'][f.type] = levels
            ca_configs_new['parameters']['range'][f.type] = [start, end]

        return ca_configs_new

    def collect_res(self, evo_step):
        if not self.res:
            self.res = {v: {} for v in evo_step.answers}

        levels = tuple(f.explict_levels for f in evo_step.factors)
        print("levels: ", levels)

        # TODO : WTF????
        ids = np.array(np.meshgrid(
            levels[0], levels[1])).T.reshape(-1, len(self.evo_params))

        data = list(evo_step.answers.values())[0]
        data = data.reshape(-1, data.shape[-1])

        verifier = list(evo_step.answers)[0]
        for i, id in enumerate(ids):
            self.res[verifier][tuple(id)] = data[i]

    # plot two factors with properties: |F| = 3
    # TODO: update plotter to accept more thatn two factors
    def plot(self):
        labels = [x for x in self.evo_params]
        ticks = {x: set() for x in self.evo_params}

        verifier = list(self.steps[0].answers)[0]
        ticks = np.array([list(x)
                         for x in self.res[verifier].keys()], dtype=np.float32)
        data = np.array(
            [x for x in self.res[verifier].values()], dtype=np.float32)

        ticks_f1 = set(ticks[:, 0].tolist())
        ticks_f2 = set(ticks[:, 1].tolist())

        labels_f1 = labels[0]
        labels_f2 = labels[1]
        print(data.shape)

        pie_scatter = PieScatter2D(data)
        pie_scatter.draw(ticks_f1, ticks_f2, labels_f1, labels_f2)
        pdf_dir = f'./pdfs_{verifier}'
        Path(pdf_dir).mkdir(parents=True, exist_ok=True)
        pie_scatter.save(f'{pdf_dir}/all.pdf')

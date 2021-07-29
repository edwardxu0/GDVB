import copy
import numpy as np

from enum import Enum, auto

from fractions import Fraction as F
from pathlib import Path

from ..core.verification_benchmark import VerificationBenchmark

from .evo_step import EvoStep

from ..plot.pie_scatter import PieScatter2D


class EvoState(Enum):
    Explore = auto()
    Refine = auto()
    Done = auto()


class EvoAction(Enum):
    Shrink = auto()
    Expand = auto()
    Empty = auto()


class EvoBench:
    def __init__(self, seed_benchmark):
        self.logger = seed_benchmark.settings.logger
        self.seed_benchmark = seed_benchmark
        self.evo_configs = seed_benchmark.settings.evolutionary
        self.evo_params = self.evo_configs['parameters']
        assert len(self.evo_params) == 2
        self._init_parameters()

    def _init_parameters(self):
        self.state = EvoState.Explore
        self.steps = []
        self.global_min = {}
        self.global_max = {}
        self.res = {}
        self.refine_iterations = 3

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
            neus = [f'{x:.3f}' for x in sorted(set([x['neu'] for x in evo_step.benchmark.ca]))]
            fcs = [f'{x:.3f}' for x in sorted(set([x['fc'] for x in evo_step.benchmark.ca]))]
            print('\n--------------------------------------------')
            print(f"Evo iter={i}, neu={neus}, fc={fcs}")
            # --

            evo_step.forward()
            evo_step.evaluate()
            evo_step.plot(i+1)
            self.collect_res(evo_step)
            self.plot(i+1)

            next_ca_configs = self.evolve(evo_step)

            if self.state == EvoState.Done:
                break

            next_benchmark = VerificationBenchmark(f'{benchmark_name}_{i}',
                                                   dnn_configs,
                                                   next_ca_configs,
                                                   evo_step.benchmark.settings)

            next_evo_step = EvoStep(
                next_benchmark, self.evo_configs['parameters'])

            self.steps += [next_evo_step]

        self.plot(i+1)
        self.logger.info('Evo bench finished successfuly!')
        print('Evo bench finished successfuly!')

    def evolve(self, evo_step):
        ca_configs = evo_step.benchmark.ca_configs

        # a decision maker: determins what actions to take in the next step?

        # A1: Exploration State
        # 1) for all factors, check if exploration is needed
        # 2) for all factors, check if configed bounds are reached
        # if 1) or 2), go to A2: Refinement state
        if self.state == EvoState.Explore:
            actions = []
            for i, f in enumerate(evo_step.factors):
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
                print(f'{f.type}: min cut={min_cut}; max cut={max_cut}.')

                if max_cut == 0:
                    action = EvoAction.Expand
                else:
                    action = EvoAction.Empty

                actions += [action]

            print(f'Evo state={self.state}, ; Actions={actions}')

            if all(x == EvoAction.Empty for x in actions):
                self.state = EvoState.Refine
                next_ca_configs = None

            elif [set(actions) & {EvoAction.Shrink, EvoAction.Expand}]:
                next_ca_configs = self.explore(evo_step, actions)

                # check if goto refine state?
                goto_refine_flags = []
                for i, p in enumerate(self.evo_params):
                    if actions[i] == EvoAction.Empty:
                        pass
                    elif actions[i] == EvoAction.Shrink:
                        this_start = F(ca_configs['parameters']['range'][p][0])
                        next_start = F(next_ca_configs['parameters']['range'][p][0])
                        goto_refine_flags += [this_start == next_start]
                    elif actions[i] == EvoAction.Expand:
                        this_end = F(ca_configs['parameters']['range'][p][1])
                        next_end = F(next_ca_configs['parameters']['range'][p][1])
                        goto_refine_flags += [this_end == next_end]
                    else:
                        raise NotImplementedError
                if all(x for x in goto_refine_flags):
                    self.state = EvoState.Refine
            else:
                raise NotImplementedError

        # A2 : Refinement State
        # 1) check starting point based on observations from the exploration states
        # 2) prune overly easy and overly hard levels
        # 3) stop when necessary, go to analyze state
        if self.state == EvoState.Refine:

            if self.refine_iterations == 3:
                next_ca_configs = self.first_refine(evo_step)
            else:
                #next_ca_configs = self.refine(evo_step)
                next_ca_configs = self.first_refine(evo_step)

            self.refine_iterations -= 1
            if self.refine_iterations == 0:
                next_ca_configs = None
                self.state = EvoState.Done
            else:
                goto_next_flags = []
                for p in self.evo_params:
                    this_range = [F(x) for x in ca_configs['parameters']['range'][p]]
                    next_range = [F(x) for x in next_ca_configs['parameters']['range'][p]]
                    this_level = ca_configs['parameters']['level'][p]
                    next_level = next_ca_configs['parameters']['level'][p]
                    goto_next_flags += [this_range == next_range and this_level == next_level]
                    print(this_level, next_level)
                    print(this_range, next_range)
                if all(x for x in goto_next_flags):
                    self.state = EvoState.Done
                print(goto_next_flags)

        if self.state == EvoState.Done:
            print('haha, Done, do some analyze now!')

        return next_ca_configs

    def explore(self, evo_step, actions):
        ca_configs = evo_step.benchmark.ca_configs
        ca_configs_new = copy.deepcopy(ca_configs)

        arity = self.evo_configs['arity']
        inflation_rate = self.evo_configs['inflation_rate']
        deflation_rate = self.evo_configs['deflation_rate']

        parameters_upper_bounds = self.evo_configs['parameters_upper_bounds']

        for i, f in enumerate(evo_step.factors):
            f = copy.deepcopy(f)
            if actions[i] == EvoAction.Shrink:
                pass
            elif actions[i] == EvoAction.Expand:
                start = f.start*inflation_rate
                end = f.end*inflation_rate

                # check bounds from evo configs
                if f.type in parameters_upper_bounds:
                    end = min(end, F(parameters_upper_bounds[f.type]))

                # skip factor-level modification if start >= end
                if start >= end:
                    continue

                f.set_start_end(start, end)

                start, end, levels = f.get()
                ca_configs_new['parameters']['level'][f.type] = levels
                ca_configs_new['parameters']['range'][f.type] = [start, end]

        return ca_configs_new

    def first_refine(self, evo_step):
        ca_configs = evo_step.benchmark.ca_configs
        ca_configs_new = copy.deepcopy(ca_configs)
        arity = self.evo_configs['arity']

        verifier = list(self.res_nb_solved)[0]
        raw = self.res_nb_solved[verifier]
        for key in raw:
            print(key, raw[key])

        for i, f in enumerate(evo_step.factors):
            f = copy.deepcopy(f)

            all_e = list(set(x[i] for x in raw))
            min_e = min(all_e)
            start = min_e

            max_e = max([x[i] for x in raw if raw[x] != i])

            tmp = [x for x in all_e if x > max_e]
            end = min(tmp) if tmp else max_e

            f.set_start_end(start, end)
            f.subdivision(arity)

            start, end, levels = f.get()
            ca_configs_new['parameters']['level'][f.type] = levels
            ca_configs_new['parameters']['range'][f.type] = [start, end]

        return ca_configs_new

    def refine(self, evo_step):
        ca_configs = evo_step.benchmark.ca_configs
        ca_configs_new = copy.deepcopy(ca_configs)

        return ca_configs_new

    def collect_res(self, evo_step):
        if not self.res:
            self.res = {v: {} for v in evo_step.answers}
            self.res_nb_solved = {v: {} for v in evo_step.answers}

        levels = tuple(f.explict_levels for f in evo_step.factors)

        # TODO : WTF???? how to separate ndarray _,_ = np.xxx(x)???
        ids = np.array(np.meshgrid(
            levels[0], levels[1])).T.reshape(-1, len(self.evo_params))

        data = list(evo_step.answers.values())[0]
        data = data.reshape(-1, data.shape[-1])

        data2 = list(evo_step.nb_solved.values())[0]
        data2 = data2.reshape(-1, 1)

        verifier = list(evo_step.answers)[0]
        for i, id in enumerate(ids):
            self.res[verifier][tuple(id)] = data[i]
            self.res_nb_solved[verifier][tuple(id)] = data2[i]

    # plot two factors with properties: |F| = 3
    # TODO: update plotter to accept more thatn two factors
    def plot(self, iteration):
        labels = [x for x in self.evo_params]
        ticks = {x: set() for x in self.evo_params}

        verifier = list(self.steps[0].answers)[0]
        ticks = np.array([list(x)
                          for x in self.res[verifier].keys()], dtype=np.float32)
        data = np.array(
            [x for x in self.res[verifier].values()], dtype=np.float32)

        ticks_f1 = ticks[:, 0].tolist()
        ticks_f2 = ticks[:, 1].tolist()

        labels_f1 = labels[0]
        labels_f2 = labels[1]

        pie_scatter = PieScatter2D(data)
        pie_scatter.draw_with_ticks(ticks_f1, ticks_f2, labels_f1, labels_f2)
        pdf_dir = f'./img/{verifier}'
        Path(pdf_dir).mkdir(parents=True, exist_ok=True)
        pie_scatter.save(f'{pdf_dir}/all_{iteration}.pdf')

        pie_scatter.draw_with_ticks(ticks_f1, ticks_f2, labels_f1, labels_f2, x_log_scale=True, y_log_scale=True)
        pie_scatter.save(f'{pdf_dir}/all_log_{iteration}.pdf')

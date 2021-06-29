import sys
import copy
import time

import numpy as np
from tqdm import tqdm

from fractions import Fraction as F

from ..verification_benchmark import VerificationBenchmark

TIME_BREAK = 10


def sampling(verification_benchmark):
    logger = verification_benchmark.settings.logger

    benchmark_evolutions = [verification_benchmark]

    benchmark_name = verification_benchmark.settings.name
    dnn_configs = verification_benchmark.settings.dnn_configs
    evo_configs = verification_benchmark.settings.evolutionary
    arity = evo_configs['arity']
    inflation_rate = evo_configs['inflation_rate']
    deflation_rate = evo_configs['deflation_rate']

    for i in range(evo_configs['iterations']):
        benchmark = benchmark_evolutions[-1]
        neus = sorted(set([x['neu'] for x in benchmark.ca]))
        fcs = sorted(set([x['fc'] for x in benchmark.ca]))
        print(i, neus, fcs, '\n')

        benchmark.train()
        nb_train_tasks = len(benchmark.verification_problems)
        progress_bar = tqdm(total=nb_train_tasks, desc="Waiting on training ... ", ascii=False, file=sys.stdout)
        while not benchmark.trained():
            progress_bar.update(benchmark.trained(True))
            time.sleep(TIME_BREAK)
        progress_bar.update(nb_train_tasks)

        benchmark.verify()
        nb_verification_tasks = len(benchmark.verification_problems)
        progress_bar = tqdm(total=nb_verification_tasks, desc="Waiting on verification ... ", ascii=False, file=sys.stdout)
        while not benchmark.verified():
            progress_bar.update(benchmark.verified(True))
            time.sleep(TIME_BREAK)
        progress_bar.update(nb_verification_tasks)

        benchmark.analyze()

        next_ca_configs = evolve(benchmark, evo_configs['parameters'], arity, inflation_rate, deflation_rate)

        next_benchmark = VerificationBenchmark(f'{benchmark_name}_{i}',
                                               dnn_configs,
                                               next_ca_configs,
                                               benchmark.settings)

        benchmark_evolutions += [next_benchmark]


def evolve(benchmark, parameters_to_change, arity, inflation_rate, deflation_rate):
    evolve_strategy, solved_per_verifiers, indexes = evaluate_benchmark(benchmark, parameters_to_change)

    a = list(solved_per_verifiers.values())[0]
    print(a)
    print(np.transpose(a))

    ca_configs = benchmark.ca_configs
    ca_configs_new = copy.deepcopy(benchmark.ca_configs)

    # 1) all too easy? inflate the benchmark
    # 2) all too hard? deflate the benchmark
    benchmark.settings.logger.debug(f'evolving ... with {evolve_strategy}')
    if evolve_strategy in ['deflate', 'inflate']:
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
        for i, param in enumerate(parameters_to_change):
            nb_levels = ca_configs['parameters']['level'][param]
            level_min = F(ca_configs['parameters']['range'][param][0])
            level_max = F(ca_configs['parameters']['range'][param][1])

            print('old',param,nb_levels, level_min, level_max)

            axes = list(range(len(parameters_to_change)))
            axes.remove(i)
            axes = [i] + axes
            raw = list(solved_per_verifiers.values())[0].transpose(axes)
            # print(raw)
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
            max_id2 = [np.all(x == 0) for x in raw].index(False)
            assert max_id == max_id2

            assert min_id is not None
            assert max_id is not None

            # print(min_id, max_id)

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

            level_min = level_min + min_id * old_step - new_step
            level_max = level_max - max_id * old_step + new_step
            nb_levels = int((level_max-level_min)/new_step + 1)

            print('new', param, level_min, level_max, new_step, nb_levels)

            ranges = [level_min, level_max]
            ca_configs_new['parameters']['level'][param] = nb_levels
            ca_configs_new['parameters']['range'][param] = ranges
    else:
        assert False

    return ca_configs_new


def evaluate_benchmark(benchmark, parameters_to_change):
    ca_configs = benchmark.ca_configs
    # prepare verification results
    indexes = {}
    for p in parameters_to_change:
        ids = []
        for vpc in benchmark.ca:
            ids += [vpc[x] for x in vpc if x == p]
        indexes[p] = sorted(set(ids))

    solved_per_verifiers = {}
    for problem in benchmark.verification_problems:
        for verifier in problem.verification_results:
            if verifier not in solved_per_verifiers:
                shape = []
                for p in parameters_to_change:
                    shape += [ca_configs['parameters']['level'][p]]
                solved_per_verifiers[verifier] = np.zeros(shape, dtype=np.int)

            idx = tuple(indexes[x].index(problem.vpc[x]) for x in parameters_to_change)
            if problem.verification_results[verifier][0] in ['sat', 'unsat']:
                solved_per_verifiers[verifier][idx] += 1

    nb_property = ca_configs['parameters']['level']['prop']
    assert len(solved_per_verifiers) == 1, 'currently only support one verifier at a time.'
    raw = list(solved_per_verifiers.values())[0]
    if np.all(raw == 0):
        evolve_strategy = 'deflate'
    elif np.all(raw == nb_property):
        evolve_strategy = 'inflate'
    else:
        evolve_strategy = 'zoom in'
    return evolve_strategy, solved_per_verifiers, indexes

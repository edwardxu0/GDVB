import sys
import time
import numpy as np

from tqdm import tqdm

from fractions import Fraction as F

from ..plot.pie_scatter import PIE_SCATTER

TIME_BREAK = 10

class EvoStep:
    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.nb_solved = None
        self.answers = None

    def proceed(self):
        self.benchmark.train()

        nb_train_tasks = len(self.benchmark.verification_problems)
        progress_bar = tqdm(total=nb_train_tasks,
                        desc="Waiting on training ... ",
                        ascii=False,
                        file=sys.stdout)
        nb_trained_pre = self.benchmark.trained(True)
        
        progress_bar.update(nb_trained_pre)
        while not self.benchmark.trained():
            time.sleep(TIME_BREAK)
            nb_trained_now = self.benchmark.trained(True)
            progress_bar.update(nb_trained_now - nb_trained_pre)
            progress_bar.refresh()
            nb_trained_pre = nb_trained_now
        progress_bar.close()

        self.benchmark.verify()

        nb_verification_tasks = len(self.benchmark.verification_problems)
        progress_bar = tqdm(total=nb_verification_tasks,
                        desc="Waiting on verification ... ",
                        ascii=False,
                        file=sys.stdout)
        
        nb_verified_pre = self.benchmark.verified(True)
        progress_bar.update(nb_verified_pre)
        while not self.benchmark.verified():
            time.sleep(TIME_BREAK)
            nb_verified_now = self.benchmark.verified(True)
            progress_bar.update(nb_verified_now - nb_verified_pre)
            progress_bar.refresh()
            nb_verified_pre = nb_verified_now
        progress_bar.close()

        self.benchmark.analyze()

    # process verification results for things
    def evaluate(self, parameters_to_change):
        benchmark = self.benchmark
        ca_configs = benchmark.ca_configs
        indexes = {}
        for p in parameters_to_change:
            ids = []
            for vpc in benchmark.ca:
                ids += [vpc[x] for x in vpc if x == p]
            indexes[p] = sorted(set(ids))

        nb_property = ca_configs['parameters']['level']['prop']
        solved_per_verifiers = {}
        answers_per_verifiers = {}
        for problem in benchmark.verification_problems:
            for verifier in problem.verification_results:
                if verifier not in solved_per_verifiers:
                    shape = ()
                    for p in parameters_to_change:
                        shape += (ca_configs['parameters']['level'][p],)
                    solved_per_verifiers[verifier] = np.zeros(shape, dtype=np.int)
                    answers_per_verifiers[verifier] = np.empty(shape+(nb_property,), dtype=np.int)
                idx = tuple(indexes[x].index(problem.vpc[x]) for x in parameters_to_change)
                if problem.verification_results[verifier][0] in ['sat', 'unsat']:
                    solved_per_verifiers[verifier][idx] += 1
                prop_id = problem.vpc['prop']
                code = benchmark.settings.answer_code[problem.verification_results[verifier][0]]
                answers_per_verifiers[verifier][idx+(prop_id,)] = code

        self.nb_solved = solved_per_verifiers
        self.answers = answers_per_verifiers


    def plot(self, factors, iteration):
        configs = self.benchmark.ca_configs['parameters']
        data = self.answers

        xlabel = factors[0]
        level_min = F(configs['range'][factors[0]][0])
        level_max = F(configs['range'][factors[0]][1])
        nb_levels = F(configs['level'][factors[0]])
        xtics = np.arange(level_min, level_min + level_max, (level_max - level_min) / (nb_levels - 1))
        xtics = [f'{float(x):.4f}' for x in xtics]
        ylabel = factors[1]
        level_min = F(configs['range'][factors[1]][0])
        level_max = F(configs['range'][factors[1]][1])
        nb_levels = F(configs['level'][factors[1]])
        ytics = np.arange(level_min, level_min + level_max, (level_max - level_min) / (nb_levels - 1))
        ytics = [f'{float(x):.4f}' for x in ytics]

        data = list(data.values())[0]
        pie_scatter = PIE_SCATTER(data)
        pie_scatter.draw(xtics, ytics, xlabel, ylabel)
        #pie_scatter.save(f'{iteration}.pdf')
        pie_scatter.save(f'a.pdf')

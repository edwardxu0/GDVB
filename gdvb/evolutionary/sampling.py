import copy
import time
import numpy as np

from ..verification_benchmark import VerificationBenchmark

TIME_BREAK = 10


def sampling(verification_benchmark):
    logger = verification_benchmark.settings.logger

    benchmark_evolutions = [verification_benchmark]

    benchmark_name = verification_benchmark.settings.name
    dnn_configs = verification_benchmark.settings.dnn_configs
    ca_configs = verification_benchmark.settings.ca_configs
    evo_configs = verification_benchmark.settings.evolutionary

    for i in range(evo_configs['iterations']):
        benchmark = benchmark_evolutions[-1]
        vpcs = [x.vpc for x in benchmark.verification_problems]
        neus = set([x['neu'] for x in vpcs])
        fcs = set([x['fc'] for x in vpcs])
        print(neus, fcs)
        benchmark.train()
        logger.info('Waiting for training ... ')
        while not benchmark.trained():
            print('.', end='')
            time.sleep(TIME_BREAK)

        # benchmark.verify()
        # logger.info('Waiting for verification ... ')
        # while not benchmark.verified():
        #     print('.', end='')
        #     time.sleep(TIME_BREAK)

        ca_configs_new = copy.copy(ca_configs)

        # if too easy: increase
        # else: reduce
        arity = evo_configs['arity']
        for evo_param in evo_configs['parameters']:
            level = ca_configs['parameters']['level'][evo_param] * arity
            ranges = [ca_configs['parameters']['range'][evo_param][0],
                      ca_configs['parameters']['range'][evo_param][1] * arity]
            ca_configs_new['parameters']['level'][evo_param] = level
            ca_configs_new['parameters']['range'][evo_param] = ranges

        next_benchmark = VerificationBenchmark(f'{benchmark_name}_{i}',
                                               dnn_configs,
                                               ca_configs,
                                               benchmark.settings)
        benchmark_evolutions += [next_benchmark]

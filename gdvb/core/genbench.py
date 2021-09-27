import random
import datetime

from .verification_benchmark import VerificationBenchmark
from ..evolutionary.evo_bench import EvoBench


# main benchmark generation function
def gen(settings):
    start_time = datetime.datetime.now()
    random.seed(settings.seed)

    verification_benchmark = VerificationBenchmark(settings.name,
                                                   settings.dnn_configs,
                                                   settings.ca_configs,
                                                   settings)

    #  perform tasks
    if settings.task == 'gen_ca':
        pass
    elif settings.task == 'train':
        verification_benchmark.train()
        # verification_benchmark.analyze_training()
    elif settings.task == 'gen_props':
        verification_benchmark.gen_props()
    elif settings.task == 'verify':
        verification_benchmark.verify()
        # verification_benchmark.analyze_verification()
    elif settings.task == 'analyze':
        verification_benchmark.analyze_all()
        verification_benchmark.save_results()
    elif settings.task == 'all':
        verification_benchmark.train()
        verification_benchmark.gen_props()
        verification_benchmark.verify()
        verification_benchmark.analyze_all()
        verification_benchmark.save_results()
    elif settings.task == 'evolutionary':
        evo_bench = EvoBench(verification_benchmark)
        evo_bench.run()
    else:
        raise Exception("Unknown task.")

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    settings.logger.info(f'Spent {duration} seconds.')

import logging
import os
import toml
import pathlib


def configure(args):
    config_file = open(args.configs, 'r').read()
    configs = toml.loads(config_file)

    configs['root'] = os.path.join(args.root, f'{configs["name"]}.{args.seed}')
    configs['seed'] = args.seed
    configs['task'] = args.task
    configs['override'] = args.override
    if args.debug:
        configs['logging_level'] = logging.DEBUG
    elif args.dumb:
        configs['logging_level'] = logging.WARN
    else:
        configs['logging_level'] = logging.INFO

    settings = Settings(configs)

    return settings


class Settings:
    def __init__(self, configs):
        self.precision = '.4f'
        self.name = configs['name']
        self.root = configs['root']
        self.seed = configs['seed']
        self.task = configs['task']
        self.logging_level = configs['logging_level']
        self.override = configs['override']

        self.dnn_configs = configs['dnn']
        self.ca_configs = configs['ca']

        self.training_configs = configs['train']
        self.verification_configs = configs['verify']
        self.evolutionary = configs['evolutionary'] if 'evolutionary' in configs else None

        self.tmp_dir = './tmp'
        pathlib.Path(self.tmp_dir).mkdir(parents=True, exist_ok=True)
        self.sub_dirs = ['dis_config', 'dis_model', 'dis_log', 'props', 'veri_log']
        if configs['train']['dispatch']['platform'] == 'slurm':
            self.sub_dirs += ['dis_slurm']
        if configs['verify']['dispatch']['platform'] == 'slurm':
            self.sub_dirs += ['veri_slurm']

        self.answer_code = {'unsat': 1,
                            'sat': 2,
                            'unknown': 3,
                            'timeout': 4,
                            'memout': 4,
                            'error': 5}
        self._make_dirs()

    def _make_dirs(self):
        pathlib.Path(self.root).mkdir(parents=True, exist_ok=True)
        pathlib.Path().mkdir(parents=True, exist_ok=True)
        for sd in self.sub_dirs:
            attr = sd + '_dir'
            self.__setattr__(attr, os.path.join(self.root, sd))
            pathlib.Path(self.__getattribute__(attr)).mkdir(parents=True, exist_ok=True)

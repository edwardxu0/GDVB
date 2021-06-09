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

    settings = Settings(configs)

    return settings


class Settings:
    def __init__(self, configs):
        self.name = configs['name']
        self.root = configs['root']
        self.seed = configs['seed']
        self.task = configs['task']
        self.override = configs['override']

        self.dnn_configs = configs['dnn']
        self.ca_configs = configs['ca']

        self.training_configs = configs['train']
        self.verification_configs = configs['verify']

        self.tmp_dir = './tmp'
        pathlib.Path(self.tmp_dir).mkdir(parents=True, exist_ok=True)
        self.sub_dirs = ['dis_config', 'dis_model', 'dis_log', 'props', 'veri_log']
        if configs['train']['dispatch']['platform'] == 'slurm':
            self.sub_dirs += ['dis_slurm']
        if configs['verify']['dispatch']['platform'] == 'slurm':
            self.sub_dirs += ['veri_slurm']

        self._make_dirs()

    def _make_dirs(self):
        pathlib.Path(self.root).mkdir(parents=True, exist_ok=True)
        pathlib.Path().mkdir(parents=True, exist_ok=True)
        for sd in self.sub_dirs:
            attr = sd + '_dir'
            self.__setattr__(attr, os.path.join(self.root, sd))
            pathlib.Path(self.__getattribute__(attr)).mkdir(parents=True, exist_ok=True)

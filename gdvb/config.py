import os
import toml
import pathlib


def configure(args):
    config_file = open(args.configs,'r').read()
    configs = toml.loads(config_file)
    configs['root'] = os.path.join(args.root,f'{configs["name"]}.{args.seed}')
    configs['seed'] = args.seed
    configs['task'] = args.task
    subdirs = ['dis_config','dis_model','dis_slurm','dis_log','props','veri_slurm','veri_log']

    pathlib.Path(configs['root']).mkdir(parents=True, exist_ok=True)
    for sd in subdirs:
        configs[sd+'_dir'] = os.path.join(configs['root'],sd)
        pathlib.Path(configs[sd+'_dir']).mkdir(parents=True, exist_ok=True)

    return configs

import os
import toml


def configure(args):
    config_file = open(args.configs,'r').read()
    configs = toml.loads(config_file)
    configs['root'] = f'{args.root}.{configs["name"]}.{args.seed}'
    #configs['root'] = args.root
    configs['seed'] = args.seed
    configs['task'] = args.task

    if not os.path.exists(configs['root']):
        os.mkdir(configs['root'])

    subdirs = ['dis_config','dis_model','dis_slurm','dis_log','dis_done','test_slurm','test_log',\
    'test_done','props','veri_net','veri_slurm','veri_log','veri_done']

    for sd in subdirs:
        configs[sd+'_dir'] = os.path.join(configs['root'],sd)
        if not os.path.exists(configs[sd+'_dir']):
            os.mkdir(configs[sd+'_dir'])

    return configs

import os
import toml


def configure(args):
    config_file = open(args.config,'r').read()
    configs = toml.loads(config_file)
    configs['root'] = args.root

    if not os.path.exists(configs['root']):
        os.mkdir(configs['root'])

    subdirs = ['dis_config_dir','dis_model_dir','dis_slurm_dir','dis_log_dir','dis_done_dir','test_slurm_dir','test_log_dir',\
    'test_done_dir','prop_dir','veri_net_dir','veri_slurm_dir','veri_log_dir','veri_done_dir']

    for sd in subdirs:
        configs[sd] = os.path.join(configs['root'],sd[:-4])
        if not os.path.exists(configs[sd]):
            os.mkdir(configs[sd])

    return configs

import argparse
from pyfiglet import Figlet


def parse_args():
    f = Figlet(font='slant')
    print(f.renderText('GDVB'), end='')
    
    parser = argparse.ArgumentParser(
        description='Generative Diverse DNN Verification Benchmarks',
        prog='GDVB')
    
    parser.add_argument('configs', type=str, help='Configurations of GDVB benchmark.')
    parser.add_argument('task', type=str, choices=['gen_ca', 'train', 'gen_props', 'verify', 'analyze', 'all', 'evolutionary'],
                        help='Select tasks to perform.')
    parser.add_argument('seed', type=int, help='Random seed.')
    parser.add_argument('--root', type=str, default='./results/', help='Root directory')
    parser.add_argument('--platform', type=str, choices=['local', 'slurm'], help='How to run jobs?')
    parser.add_argument('--override', action='store_true', help='Override existing logs?')
    parser.add_argument('--debug', action='store_true', help='Verbose mode.')

    return parser.parse_args()

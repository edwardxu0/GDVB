import argparse
from pyfiglet import Figlet


def parse_args():
    f = Figlet(font='slant')
    print(f.renderText('GDVB'), end='')

    parser = argparse.ArgumentParser(
        description='Generative Diverse DNN Verification Benchmarks',
        prog='GDVB')

    parser.add_argument('configs', type=str,
                        help='Configurations file.')
    parser.add_argument('task', type=str,
                        choices=['gen_ca', 'train', 'gen_props', 'verify', 'analyze', 'all'],
                        help='Select tasks to perform.')
    parser.add_argument('--seed', type=int,
                        default=0,
                        help='Random seed.')
    parser.add_argument('--result_dir', type=str,
                        default='./results/',
                        help='Root directory.')
    parser.add_argument('--platform', type=str,
                        default='local',
                        choices=['local', 'slurm'],
                        help='Platform to run jobs.')
    parser.add_argument('--override', action='store_true',
                        help='Override existing logs.')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug log.')
    parser.add_argument('--dumb', action='store_true',
                        help='Silent mode.')

    return parser.parse_args()

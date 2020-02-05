import argparse
from pyfiglet import Figlet

def _parse_args():
    f = Figlet(font='slant')
    print(f.renderText('GenBench'), end='')
    
    parser = argparse.ArgumentParser(
        description='Neural Architecture Search',
        prog='GenBench')
    
    parser.add_argument('configs', type=str, help='configs')
    parser.add_argument('task', type=str, help='task')
    parser.add_argument('seed', type=int)
    parser.add_argument('--root', type=str, default='./results/',help='root directory')

    return parser.parse_args()

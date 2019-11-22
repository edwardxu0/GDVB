import argparse
from pyfiglet import Figlet

def _parse_args():
    f = Figlet(font='slant')
    print(f.renderText('GenBench'), end='')
    
    parser = argparse.ArgumentParser(
        description='Neural Architecture Search',
        prog='GenBench')
    
    parser.add_argument('config', type=str, help='NAS config')
    parser.add_argument('--root', type=str, default='./res',help='root directory')


    return parser.parse_args()

"""
"""
import argparse
import random

from . import cli
from . import logging


def main(args: argparse.Namespace):
    logger = logging.initialize(__package__, args)
    random.seed(args.seed)
    args.func(args)


def __main__():
    return main(cli.parse_args())


if __name__ == "__main__":
    __main__()

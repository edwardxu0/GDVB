from . import cli
from . import config
from . import genbench
from . import logging


def main():
    args = cli.parse_args()
    configs = config.configure(args)
    logging.initialize(configs)
    genbench.gen(configs)


if __name__ == '__main__':
    main()

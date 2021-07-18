from .misc import cli
from .misc import config
from .misc import logging
from .core import genbench


def main():
    args = cli.parse_args()
    configs = config.configure(args)
    logging.initialize(configs)
    genbench.gen(configs)


if __name__ == '__main__':
    main()

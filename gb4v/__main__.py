from . import cli
from . import config
from . import genbench
from . import logging


def main():
    args = cli._parse_args()
    configs = config.configure(args)
    logger = logging.initialize(args)
    configs['logger'] = logger
    #genbench.gen(args, configs)
    genbench.gen2(args, configs)

if __name__ == '__main__':
    main()

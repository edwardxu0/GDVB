import sys
import logging

logging.basicConfig(stream=sys.stdout,
                       level=logging.DEBUG,
                       format='%(asctime)s %(message)s',
                       datefmt='%m/%d/%Y %I:%M:%S %p')

def initialize(args):
    logger = logging.getLogger()

    return logger

import sys
import logging

logging.basicConfig(stream=sys.stdout,
                       format='%(asctime)s %(message)s',
                       datefmt='%m/%d/%Y %I:%M:%S %p')

def initialize(args):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    return logger

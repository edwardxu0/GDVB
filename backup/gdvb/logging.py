import sys
import logging

logging.basicConfig(stream=sys.stdout,
                       format='%(asctime)s %(message)s',
                       datefmt='%m/%d/%Y %I:%M:%S %p :')

def initialize(configs):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    configs['logger'] = logger

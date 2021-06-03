import sys
import logging

logging.basicConfig(stream=sys.stdout,
                    format='[%(levelname)s] %(asctime)s %(message)s ',
                    datefmt='%m/%d/%Y %I:%M:%S %p :')


def initialize(settings):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    settings.logger = logger

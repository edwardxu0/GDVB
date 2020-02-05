import os
import sys
import random
import argparse
import numpy as np
from functools import partial
from PIL import Image




class GeneralNormalizer():

    def __init__(self, data, min_, max_):
        self._data = data
        self._min = min_
        self._max = max_
        self._range = max_ - min_
        self._normalized = False


class ZeroOneNormalizer(GeneralNormalizer):

    def  __init__(self, data, min_, max_):
        super(data, min_, max_)

    def normalize_input(self):
        assert self._normalized == False

        norm_data = []
        for x in self._data:
            norm_data += [(x-self._min)/self._range]
        
        self._data = norm_data
        self._normalized = True



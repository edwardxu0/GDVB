import numpy as np

from fractions import Fraction as F


class Factor:
    # definition attrribute:
    # 1. defines the level coverage
    # 2. can be changed
    start = None
    end = None
    levels = None

    # innert attributes:
    # 1. defines the limitation of self
    # 2. cannot change
    type = None
    min_step = None

    # inffered attributes:
    # 1. inffered variables based on definition
    # 2. changes when definition attributes changes
    step = None

    def __init__(self, type, start, end, levels, fc_conv_ids):
        # definition attributes
        self.start = F(start)
        self.end = F(end)
        self.levels = int(levels)

        # innert attributes
        self.type = type
        if self.type in ['fc', 'conv']:
            self.min_step = F(1)/F(len(fc_conv_ids[self.type]))
        self._workout()

    def _check_min_step(self):
        # inffered attribute
        self.step = (self.end - self.start)/(self.levels - 1)
        if self.min_step and self.start < self.min_step:
            self.start = self.min_step
        if self.min_step and self.step < self.min_step:
            self.step = self.min_step

    def _workout(self):
        self._check_min_step()
        self.explict = np.arange(self.start, self.end + self.step, self.step)
        assert(self.levels == len(self.explict))

    # scale
    def scale(self, coefficient):
        self.start = self.start * coefficient
        self.end = self.end * coefficient
        self._workout()

    def set_start(self, new_start):
        self.start = new_start
        self._workout()

    def set_end(self, new_end):
        self.end = new_end
        self._workout()

    def set_start_end(self, new_start, new_end):
        self.start = new_start
        self.end = new_end
        self._workout()

    def subdivision(self, arity):
        self.levels = int(self.levels * F(arity))
        self._workout()

    def get(self):
        return self.start, self.end, self.levels

    def __str__(self):
        res = f'{self.type} : ['
        assert len(self.explict) > 0
        for x in self.explict:
            res += f'{x}, '
        res = res[:-2] + ']'
        return res

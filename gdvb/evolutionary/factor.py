import numpy as np

from fractions import Fraction as F


class Factor:
    # definition attribute:
    # 1. defines the level coverage
    # 2. can be changed
    start = None
    end = None
    nb_levels = None

    # innert attributes:
    # 1. defines the limitation of self
    # 2. cannot change
    type = None
    min_step = None

    # inferred attributes:
    # 1. inferred variables based on definition
    # 2. changes when definition attributes changes
    step = None

    def __init__(self, type, start, end, nb_levels, fc_conv_ids):
        # definition attributes
        self.start = F(start)
        self.end = F(end)
        self.nb_levels = int(nb_levels)

        # innert attributes
        self.type = type
        if self.type in ['fc', 'conv']:
            self.min_step = F(1)/F(len(fc_conv_ids[self.type]))
        self._workout()

    def _check_min_step(self):
        # inferred attribute
        self.step = (self.end - self.start)/(self.nb_levels - 1)
        if self.min_step and self.start < self.min_step:
            self.start = self.min_step
        if self.min_step and self.step < self.min_step:
            self.step = self.min_step

    def _workout(self):
        self._check_min_step()
        self.explict_levels = np.arange(self.start,
                                        self.end + self.step, self.step)
        assert(self.nb_levels == len(self.explict_levels)
               ), f"{self.nb_levels}/{self.explict_levels}"

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
        self.nb_levels = int(self.nb_levels * F(arity))
        self._workout()

    def get(self):
        return self.start, self.end, self.nb_levels

    def __str__(self):
        res = f'{self.type} : ['
        assert len(self.explict_levels) > 0
        for x in self.explict_levels:
            res += f'{x}, '
        res = res[:-2] + ']'
        return res

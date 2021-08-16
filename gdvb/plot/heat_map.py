from . import PLOT

from matplotlib import colors

import matplotlib.pyplot as plt
import seaborn as sns


class HeatMap(PLOT):
    def __init__(self, data):
        self.data = data
        self.plt = plt

    def draw(self):
        super().draw()

        M = self.data.max()

        fig, ax = plt.subplots(1, 1, figsize=(50, 8))

        cmap = colors.ListedColormap(['yellow', 'blue'])

        sns.heatmap(self.data, cmap=cmap, square=True, linewidth=0.1, linecolor='white',
                    ax=ax, cbar_kws={"shrink": 0.6, 'pad': 0.01})

        plt.show()

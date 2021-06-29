import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd
import seaborn as sns
import pathlib

import matplotlib.colors as colors
from matplotlib.patches import Patch

from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker

fontP = FontProperties()
root = './data'

from gdvb.plot import PLOT

class PIE_SCATTER(PLOT):
    def __init__(self, data):
        self.data = data
        self.plt = plt

    # https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
    def draw_pie(self, dist, xpos, ypos, size, colors, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # for incremental pie slices
        cumsum = np.cumsum(dist)
        cumsum = cumsum / cumsum[-1]
        pie = [0] + cumsum.tolist()

        markers = []
        for c, r1, r2 in zip(colors, pie[:-1], pie[1:]):
            if r1 == r2:
                continue

            angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)

            x = [0] + np.cos(angles).tolist()
            y = [0] + np.sin(angles).tolist()

            xy = np.column_stack([x, y])

            markers.append({'marker': xy, 's': size, 'facecolor': c})

            # ax.scatter([xpos], [ypos], marker=markers, s=size,)
        for marker in markers:
            ax.scatter(xpos, ypos, **marker)

        return ax

    def update_ticks(self, x, pos):
        ticks = (np.arange(0.05, 1.05, 0.05) * 784).tolist()
        return ticks[pos]

    def draw(self, xtics, ytics, xlabel, ylabel):

        levels = self.data.shape

        answers_pool = ['unsat', 'sat', 'unknown', 'oor', 'error']
        # color_list = sns.color_palette()
        color_list = ['lightgreen', 'deepskyblue', 'yellow', 'darkorange', 'crimson']

        # make legend
        legend_elements = []
        for i, ans in enumerate(answers_pool):
            le = Patch(facecolor=color_list[i], edgecolor='r', label=ans)
            legend_elements += [le]

        fig, ax = plt.subplots(figsize=(levels[0], levels[1]))
        fontP.set_size(20)

        X = []
        Y = []
        for i in range(levels[0]):
            for j in range(levels[1]):
                Z = []
                X += [i + 1]
                Y += [j + 1]
                for k in range(1, 6):
                    Z += [len(np.where(k == self.data[i][j])[0])]
                self.draw_pie(Z, [i + 1], [j + 1], 2200, color_list, ax=ax)

        # plt.legend(handles=legend_elements, bbox_to_anchor=(2, 1), loc = 'upper right', prop=fontP)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        #plt.xlabel('nueron', fontsize=20)
        plt.xlabel(xlabel)
        #plt.xticks(np.arange(0,levels[0]+1).tolist(),fontsize=16)
        print(len(np.arange(0,levels[0]+1).tolist()), len([0]+xtics))
        plt.xticks(np.arange(0,levels[0]+1).tolist(),[0]+xtics)
        plt.xlim(0, levels[0] + 1)
        #plt.ylabel('FC', fontsize=20)
        plt.ylabel(ylabel)
        #plt.yticks(np.arange(0,levels[1]+1).tolist(),fontsize=16)
        plt.yticks(np.arange(0,levels[1]+1).tolist(),[0]+ytics)
        plt.ylim(0, levels[1] + 1)
        #plt.title('title', fontsize=40)
        #plt.savefig(f'{res_dir}/{v}.png', bbox_inches='tight')
        #plt.show()

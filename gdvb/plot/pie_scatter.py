from gdvb.plot import PLOT
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch

from matplotlib.font_manager import FontProperties

fontP = FontProperties()

answers_pool = ["unsat", "sat", "unknown", "oor", "error", "hardware"]
color_list = ["lightgreen", "deepskyblue", "yellow", "darkorange", "crimson", "grey"]


class PieScatter2D(PLOT):
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

            markers.append({"marker": xy, "s": size, "facecolor": c})

            # ax.scatter([xpos], [ypos], marker=markers, s=size,)
        for marker in markers:
            ax.scatter(xpos, ypos, **marker)

        return ax

    def draw(self, xtics, ytics, xlabel, ylabel, title=False, rotation=(0, 0)):
        levels = self.data.shape

        # make legend
        legend_elements = []
        for i, ans in enumerate(answers_pool):
            le = Patch(facecolor=color_list[i], edgecolor="r", label=ans)
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
                for k in range(1, len(color_list) + 1):
                    Z += [len(np.where(k == self.data[i][j])[0])]
                print(Z)
                self.draw_pie(Z, [i + 1], [j + 1], 2200, color_list, ax=ax)

        # plt.legend(handles=legend_elements, bbox_to_anchor=(2, 1), loc = 'upper right', prop=fontP)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        # plt.xlabel('nueron', fontsize=20)
        plt.xlabel(xlabel)
        # plt.xticks(np.arange(0,levels[0]+1).tolist(),fontsize=16)
        plt.xticks(
            np.arange(0, levels[0] + 1).tolist(), [0] + xtics, rotation=rotation[0]
        )

        plt.xlim(0, levels[0] + 1)
        # plt.ylabel('FC', fontsize=20)
        plt.ylabel(ylabel)
        # plt.yticks(np.arange(0,levels[1]+1).tolist(),fontsize=16)
        plt.yticks(
            np.arange(0, levels[1] + 1).tolist(), [0] + ytics, rotation=rotation[1]
        )
        plt.ylim(0, levels[1] + 1)

        if title:
            plt.title(title)

        # plt.savefig(f'{res_dir}/{v}.png', bbox_inches='tight')
        # plt.show()

    def draw_with_ticks(
        self,
        xticks,
        yticks,
        xlabel,
        ylabel,
        x_log_scale=False,
        y_log_scale=False,
        pie_size=1000,
    ):
        assert len(xticks) == len(yticks)
        assert len(xticks) == len(self.data)

        # make legend
        legend_elements = []
        for i, ans in enumerate(answers_pool):
            le = Patch(facecolor=color_list[i], edgecolor="r", label=ans)
            legend_elements += [le]

        size_x = len(set(xticks)) + 1
        size_y = len(set(yticks)) + 1

        fig, ax = plt.subplots(figsize=((size_x, size_y)))
        fontP.set_size(20)

        for i, raw in enumerate(self.data):
            Z = []
            for k in range(1, len(color_list) + 1):
                Z += [len(np.where(k == raw)[0])]
            loc_x = xticks[i]
            loc_y = yticks[i]
            self.draw_pie(Z, loc_x, loc_y, pie_size, color_list, ax=ax)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.get_xaxis().set_major_formatter(FormatStrFormatter("%.2f"))
        ax.get_yaxis().set_major_formatter(FormatStrFormatter("%.2f"))

        plt.xlabel(xlabel)
        # plt.xticks(np.arange(0, levels[0]+1).tolist(), [0]+xticks)
        plt.xticks(list(set(xticks)))
        if x_log_scale:
            plt.xscale("log")
        else:
            plt.xlim(0, max(xticks) * (1 + 1 / size_x))

        plt.ylabel(ylabel)
        # plt.yticks(np.arange(0, levels[1]+1).tolist(), [0]+yticks)
        plt.yticks(list(set(yticks)))

        if y_log_scale:
            plt.yscale("log")
        else:
            plt.ylim(0, max(yticks) * (1 + 1 / size_y))

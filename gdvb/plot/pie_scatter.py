from gdvb.plot import PLOT
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch

from matplotlib.font_manager import FontProperties
from matplotlib import rc

# activate latex text rendering
rc("text", usetex=True)


fontP = FontProperties()


answers_pool = ["unsat", "sat", "unknown", "oor", "error", "hardware"]
legend_labels = [
    "Unsat",
    "Sat",
    "Unknown",
    "Out of resource",
    "Error",
    "Model not trained\n due to GPU RAM limit",
]
color_list = ["lightgreen", "deepskyblue", "yellow", "darkorange", "crimson", "grey"]

label_code = {
    "neu": r"\textbf{Factor} \textit{level} scale($\times$): Neurons",
    "fc": r"\textbf{Factor} \textit{level} scale($\times$): FC Layers",
    "conv": r"\textbf{Factor} \textit{level} scale($\times$): Conv Layers",
    "eps": r"\textbf{Factor} \textit{level} scale($\times$): Radii($\epsilon$)",
}


class PieScatter2D(PLOT):
    def __init__(self, data):
        self.data = data
        self.plt = plt

    # https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
    def draw_pie(self, dist, xpos, ypos, size, colors, ax=None, highlight=None):
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
            # print(xpos, ypos)
            # exit()
            if highlight:
                ax.scatter(xpos, ypos, s=size, facecolor="none", edgecolor="r")

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
                # print(Z)
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
        tick_size=16,
        label_size=20,
        legend_size=10,
        display_legend=False,
    ):
        assert len(xticks) == len(yticks)
        assert len(xticks) == len(self.data)

        # make legend

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

            # TODO: fix this hard-coded for paper writing
            highlight = [xticks[i], yticks[i]] in [
                [7.636363506317139, 5.090909004211426],
                [13.090909004211426, 5.090909004211426],
                [16.727272033691406, 5.090909004211426],
                [24.0, 5.090909004211426],
                [9.454545021057129, 12.606060981750488],
                [16.727272033691406, 12.606060981750488],
                [24.0, 27.636363983154297],
            ]
            self.draw_pie(
                Z, loc_x, loc_y, pie_size, color_list, ax=ax, highlight=highlight
            )

        ax.get_xaxis().set_major_formatter(FormatStrFormatter("%.2f"))
        ax.get_yaxis().set_major_formatter(FormatStrFormatter("%.2f"))

        plt.xlabel(label_code[xlabel], fontsize=label_size)
        # plt.xticks(np.arange(0, levels[0]+1).tolist(), [0]+xticks)
        plt.xticks(list(set(xticks)), fontsize=tick_size)
        if x_log_scale:
            plt.xscale("log")
        else:
            pass
            # min_ = min(xticks) * (1 - 1 / size_x)
            # max_ = max(xticks) * (1 + 1 / size_x)
            # plt.xlim(min_, max_ )

        plt.ylabel(label_code[ylabel], fontsize=label_size)
        # plt.yticks(np.arange(0, levels[1]+1).tolist(), [0]+yticks)
        plt.yticks(list(set(yticks)), fontsize=tick_size)

        if display_legend:
            legend_elements = []
            for i in range(len(color_list)):
                le = Patch(
                    facecolor=color_list[i], edgecolor="black", label=legend_labels[i]
                )
                legend_elements += [le]
            ax.legend(handles=legend_elements, loc="upper left", fontsize=legend_size)
        elif ax.get_legend() is not None:
            ax.get_legend().remove()

        if y_log_scale:
            plt.yscale("log")
        else:
            pass
            # min_ = min(yticks) * (1 - 1 / size_y)
            # max_ = max(yticks) * (1 + 1 / size_y)
            # plt.ylim(min_, max_ )

    def heatmap(self, xticks, yticks, xlabel, ylabel, tick_size=16, label_size=20):
        ax = sns.heatmap(self.data, linewidth=0.5)
        plt.xlabel(label_code[xlabel], fontsize=label_size)
        # plt.xticks(np.arange(0, levels[0]+1).tolist(), [0]+xticks)
        plt.xticks(list(set(xticks)), fontsize=tick_size)
        plt.ylabel(label_code[ylabel], fontsize=label_size)
        # plt.yticks(np.arange(0, levels[1]+1).tolist(), [0]+yticks)
        plt.yticks(list(set(yticks)), fontsize=tick_size)

import os

import matplotlib

TIME_DAYS_LABEL = "Time, days"
PRICE_USD_LABEL = "Price, USD"


def set_legend(ax):
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.2))
    legend.get_frame().set_alpha(0.5)


def save_image(plt, title):
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.serif"] = "courier"
    matplotlib.rcParams["font.size"] = 11

    path = os.path.join('..', 'data', 'graphs', title.replace(' ', '_').replace(',', ''))
    plt.savefig(f"{path}.pdf", bbox_inches='tight')


def set_ticks_size(ax, axis, size):
    ax[0].tick_params(axis=axis, labelsize=size)
    ax[1].tick_params(axis=axis, labelsize=size)
    ax[2].tick_params(axis=axis, labelsize=size)

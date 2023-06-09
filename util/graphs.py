import os

import matplotlib
import matplotlib.pyplot as plt

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


def plot_series(stock, title: str, **kwargs) -> None:
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, axisbelow=True)
    for label, series in (kwargs.items()):
        ax.plot(series.values, markersize=1.0, label=label)
    title = f"{stock.company_name} {title}"
    ax.set_title(title)
    ax.set_xlabel(TIME_DAYS_LABEL)
    ax.set_ylabel(PRICE_USD_LABEL)
    set_legend(ax)
    save_image(plt, title)
    plt.show()


def plot_returns(data_to_learn_and_validate):
    plt.figure(figsize=(10, 4))
    plt.plot(data_to_learn_and_validate)
    plt.ylabel('Return', fontsize=20)


def plot_pacf(data_to_learn):
    plot_pacf(data_to_learn)
    plt.show()


def plot_acf(data_to_learn):
    plot_acf(data_to_learn)
    plt.show()

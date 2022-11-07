import numpy as np 
import matplotlib.pyplot as plt



def plot_eci_uncertainty(eci: np.ndarray, title=False) -> plt.figure:
    """
    Parameters
    ----------
    eci: numpy.array
        nxm matrix of ECI values: m sets of n ECI

    Returns
    -------
    fig: matplotlib.pyplot figure
    """
    stats = np.array([[np.mean(row), np.std(row)] for row in eci])
    print(stats.shape)
    means = stats[:, 0]
    std = stats[:, 1]
    index = list(range(eci.shape[0]))

    plt.scatter(index, means, color="xkcd:crimson", label="Means")
    plt.errorbar(index, means, std, ls="none", color="k", label="Stddev")

    plt.xlabel("ECI index (arbitrary)", fontsize=22)
    plt.ylabel("ECI magnitude (eV)", fontsize=22)
    if title:
        plt.title(title, fontsize=30)
    else:
        plt.title("STAN ECI", fontsize=30)
        plt.legend(fontsize=21)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)

    return fig


def plot_eci_hist(eci_data, xmin=None, xmax=None):
    plt.hist(x=eci_data, bins="auto", color="xkcd:crimson", alpha=0.7, rwidth=0.85)
    if xmin and xmax:
        plt.xlim(xmin, xmax)
    plt.xlabel("ECI value (eV)", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    fig = plt.gcf()
    return fig



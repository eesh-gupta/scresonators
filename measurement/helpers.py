import numpy as np
import scipy.constants as cs
import seaborn as sns
import matplotlib.pyplot as plt

def n(p, f, q, qc):
    return pow_res(p) * q**2 / qc / (cs.h * f**2 * np.pi)


def pow_res(p):
    return 10 ** (p / 10) * 1e-3

def get_homophase(config):
    """
    Calculate the list of frequencies that gives you equal phase spacing
    Parameters:
    config (dict): A dictionary containing the following keys:
        - "npoints" (int): Number of points in the frequency list.
        - "span" (float): Frequency span.
        - "kappa" (float): linewidth
        - "kappa_inc" (float): expected linewidth fudge factor (fix me).
        - "center_freq" (float): Center frequency.
    Returns:
    numpy.ndarray: An array containing the calculated frequency list.
    """
    nlin = 2

    N = config["npoints"] - nlin * 2
    df = config["span"]
    w = df / config["kappa"] * config["kappa_inc"]
    at = np.arctan(2 * w / (1 - w**2)) + np.pi
    R = w / np.tan(at / 2)
    fr = config["freq_center"]
    n = np.arange(N) - N / 2 + 1 / 2
    flist = fr + R * df / (2 * w) * np.tan(n / (N - 1) * at)
    flist_lin = (
        -np.arange(nlin, 0, -1) * df / N * 3
        + config["freq_center"]
        - config["span"] / 2
    )
    flist_linp = (
        np.arange(1, nlin + 1) * df / N * 3 + config["freq_center"] + config["span"] / 2
    )
    flist = np.concatenate([flist_lin, flist, flist_linp])
    return flist

def config_figs():

    # Set seaborn color palette
    colors = ["#0869c8", "#b51d14", '#ddb310', '#658b38', '#7e1e9c', '#75bbfd', '#cacaca']
    sns.set_palette(sns.color_palette(colors))

    # Figure parameters
    plt.rcParams['figure.figsize'] = [8, 4]
    plt.rcParams.update({'font.size': 13})
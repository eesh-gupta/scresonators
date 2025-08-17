Tc_nb = 9.288
Tc_Ta = 4.48
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cs
import seaborn as sns
from scipy import special
from scipy.optimize import curve_fit
import traceback

colors = ["#4053d3", "#b51d14", "#ddb310", "#658b38", "#7e1e9c", "#75bbfd", "#cacaca"]


# Go from power in dBm to power in W
def pow_res(p, atten):
    return 10 ** ((p + atten) / 10) * 1e-3


# Photons
def n(p, f, q, qc, atten):
    return pow_res(p, atten) * q**2 / qc / (cs.h * f**2 * np.pi)


# Boltzmann
def tp(f, T):
    return np.tanh(cs.h * f / (2 * cs.k * T))


# T: temperature, nc: critical phonon number, f: frequency, Qtls0: TLS limit, beta: power law, n: photon number
def Qtls(n, T, f, Qtls0, nc, beta):
    return Qtls0 / tp(f, T) * np.sqrt(1 + (n / nc) ** beta * tp(f, T))


# MB fit; Quasiparticle quality
def Qqp(T, f, Qqp0, Tc):
    return (
        Qqp0
        * np.exp(1.764 * Tc / T)
        / np.sinh(cs.h * f / 2 / cs.k / T)
        / special.kn(0, cs.h * f / 2 / cs.k / T)
    )


# Quality including TLS, QP, other
def Qtot(n, T, f, Qqp0, Qtls0, Qoth, Tc, beta, nc):
    return 1 / (1 / Qqp(T, f, Qqp0, Tc) + 1 / Qtls(n, T, f, Qtls0, beta, nc) + 1 / Qoth)


# Quality including TLS and other
def Qtotn(n, T, f, Qtls0, Qoth, nc, beta):
    return 1 / (1 / Qtls(n, T, f, Qtls0, nc, beta) + 1 / Qoth)


def Gamma_tot(n, T, f, Qtls0, Qoth, nc, beta):
    return 1 / Qtls(n, T, f, Qtls0, nc, beta) + 1 / Qoth


# Houck lab TLS model
def Qtls2(n, T, f, Qtls0, b1, b2, D):
    return Qtls0 * np.sqrt(1 + n**b2 / (D * T**b1) * tp(f, T) / tp(f, T))


def get_photons(res_params, cfg):
    """
    Calculate the photon number for each resonance in res_params.
    """

    for i in range(len(res_params)):
        nn = n(
            res_params[i]["pow"][0, :],
            res_params[i]["freqs"][0, :],
            res_params[i]["q"][0, :],
            res_params[i]["qc"][0, :],
            cfg["atten"],
        )
        res_params[i]["nphotons"] = nn
    return res_params


def fit_qi(
    res_params,
    cfg,
    base_pth,
    min_power_vec=None,
    max_power_vec=None,
    name=None,
    bounds=([0, 0, 0, 0], [1e8, 3e7, 1e6, 5]),
):

    j = 0  # Temperature
    params_list = []
    plt.rcParams["lines.markersize"] = 6
    min_power = cfg["min_power"]
    max_power = cfg["max_power"]
    if name is None:
        name = cfg["meas"][0]

    fig, ax = plt.subplots(3, 3, figsize=(10, 9))
    ax = ax.flatten()
    err_list = []
    qi_0, qi_hi, nn_min, nn_max = [], [], [], []
    for i in range(len(res_params)):

        # photon numbers
        nn = res_params[i]["nphotons"][:]
        freq = res_params[i]["freqs"][j, 0]
        nn_min.append(np.min(nn))
        nn_max.append(np.max(nn))
        # Assume we know the temp and freq
        q_fit_tls = lambda n, Qtls0, nc, beta: Qtls(
            n, cfg["temp"], freq, Qtls0, nc, beta
        )
        q_fitn = lambda n, Qtls0, Qoth, nc, beta: Qtotn(
            n, cfg["temp"], freq, Qtls0, Qoth, nc, beta
        )

        gamma_fitn = lambda n, Qtls0, Qoth, nc, beta: Gamma_tot(
            n, cfg["temp"], freq, Qtls0, Qoth, nc, beta
        )
        if min_power_vec is not None:
            min_power = min_power_vec[i]
        if max_power_vec is not None:
            max_power = max_power_vec[i]
        inds = np.where(
            (res_params[i]["pow"][j, :] >= min_power)
            & (res_params[i]["pow"][j, :] <= max_power)
        )
        nn_fit = nn[inds]
        qi_fit = res_params[i]["qi"][j, :][inds]
        qi_err = res_params[i]["qi_err"][j, :][inds]
        # print(nn_fit)
        p = [np.min(qi_fit), np.max(qi_fit), 3, 0.4]
        use_gamma = False
        try:

            if use_gamma:
                p, err = curve_fit(
                    gamma_fitn,
                    nn_fit,
                    1 / qi_fit,
                    p0=p,
                    sigma=qi_err / qi_fit**2,
                    bounds=bounds,
                )
                err = np.sqrt(np.diag(err))
                ax[i].semilogx(nn_fit, gamma_fitn(nn_fit, *p) * 1e6, "-", linewidth=1)
                qi_0.append(1 / gamma_fitn(0, *p))
            else:
                p, err = curve_fit(
                    q_fitn, nn_fit, qi_fit, p0=p, sigma=qi_err, bounds=bounds
                )
                err = np.sqrt(np.diag(err))
                ax[i].semilogx(nn_fit, q_fitn(nn_fit, *p) / 1e6, "-", linewidth=1)
                qi_0.append(q_fitn(0, *p))

        except:
            err = np.nan * np.ones(4)
            qi_0.append(np.min(qi_fit))
            p = np.nan * np.ones(4)
            print("Failed!")
        qi_hi.append(np.max(qi_fit))
        if not use_gamma:
            ax[i].text(
                0.1,
                0.9,
                str(cfg["pitch"][i]) + " µm",
                transform=ax[i].transAxes,
                fontsize=12,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", edgecolor="black"),
            )
        else:
            ax[i].text(
                0.3,
                0.1,
                str(cfg["pitch"][i]) + " µm",
                transform=ax[i].transAxes,
                fontsize=12,
                va="bottom",
                ha="right",
                bbox=dict(facecolor="white", edgecolor="black"),
            )

        err_list.append(err)
        if use_gamma:
            ax[i].errorbar(
                nn_fit,
                1 / (qi_fit / 1e6),
                yerr=qi_err / qi_fit**2 * 1e6,
                fmt=".",
                color=colors[1],
                label=cfg["pitch"][i],
            )
        else:
            ax[i].errorbar(
                nn_fit,
                qi_fit / 1e6,
                yerr=qi_err / 1e6,
                fmt=".",
                color=colors[1],
                label=cfg["pitch"][i],
            )
        ax[i].set_xscale("log")
        params_list.append(p)
    for a in ax:
        a.set_xlabel(r"$\langle n \rangle$")
        a.set_ylabel(r"$Q_i \: (10^6)$")

    fig.tight_layout()
    try:
        fig.savefig(base_pth + name + "_qi.png", dpi=300)
    except:
        pass

    cfg["qtls0"] = np.array([params_list[i][0] for i in range(len(params_list))])
    cfg["qother"] = np.array([params_list[i][1] for i in range(len(params_list))])
    cfg["nc"] = np.array([params_list[i][2] for i in range(len(params_list))])
    cfg["beta"] = np.array([params_list[i][3] for i in range(len(params_list))])

    cfg["qtls0_err"] = np.array([err_list[i][0] for i in range(len(err_list))])
    cfg["qother_err"] = np.array([err_list[i][1] for i in range(len(err_list))])
    cfg["nc_err"] = np.array([err_list[i][2] for i in range(len(err_list))])
    cfg["beta_err"] = np.array([err_list[i][3] for i in range(len(err_list))])
    cfg["qi0"] = np.array(qi_0)
    cfg["qi_hi"] = np.array(qi_hi)
    cfg["nn_min"] = np.array(nn_min)
    cfg["nn_max"] = np.array(nn_max)

    return cfg


def _fit_qi_model(nn_fit, qi_fit, qi_err, fit_func, bounds):
    """Helper function to fit Qi data."""
    initial_guess = [np.min(qi_fit), np.max(qi_fit), 3, 0.4]
    try:
        p, err = curve_fit(
            fit_func, nn_fit, qi_fit, p0=initial_guess, sigma=qi_err, bounds=bounds
        )
        return p, np.sqrt(np.diag(err))
    except Exception as e:
        print(f"Fitting failed: {e}")
        traceback.print_exc()
        return np.nan * np.ones(4), np.nan * np.ones(4)


def _plot_qi_fit(
    ax,
    nn_fit,
    qi_fit,
    qi_err,
    fit_params,
    fit_func,
    use_gamma,
    pitch_label,
    nn,
    qi,
    show_all,
    wide=False,
    Qtls_func=None,
):
    """Helper function to plot Qi fit."""
    if wide:
        nn_line = np.logspace(
            np.log10(np.min(nn_fit) * 0.1), np.log10(10 * np.max(nn_fit)), 350
        )
    else:
        nn_line = np.logspace(np.log10(np.min(nn_fit)), np.log10(np.max(nn_fit)), 300)
    # nn_line = nn_fit
    if use_gamma:
        y_data = 1 / (qi_fit / 1e6)
        y_data_raw = 1 / (qi / 1e6)
        y_err_data = qi_err / qi_fit**2 * 1e6
        fit_line = fit_func(nn_line, *fit_params) * 1e6

        text_pos = (0.3, 0.1)
        va, ha = "bottom", "right"
    else:
        y_data = qi_fit / 1e6
        y_data_raw = qi / 1e6
        y_err_data = qi_err / 1e6
        fit_line = fit_func(nn_line, *fit_params) / 1e6
        fit_line2 = (
            Qtls_func(nn_line, fit_params[0], fit_params[2], fit_params[3]) / 1e6
            if Qtls_func
            else None
        )
        text_pos = (0.1, 0.9)
        va, ha = "top", "left"

    # ax.axhline(
    #     fit_params[0] / 1e6, color="gray", linestyle="--", linewidth=1, label="Qi0"
    # )
    ax.axhline(
        fit_params[1] / 1e6, color="black", linestyle="--", linewidth=1, label="Qoth"
    )

    ax.errorbar(
        nn_fit, y_data, yerr=y_err_data, fmt=".", color=colors[1], label=pitch_label
    )
    if show_all:
        ax.plot(nn, qi / 1e6, ".k", label="Raw data")
    ax.semilogx(nn_line, fit_line, "-", linewidth=1)
    # ax.semilogx(nn_line, fit_line2, "-", linewidth=1)
    # ax.set_ylim([np.nanmin(qi_fit / 1e6) * 0.8, np.nanmax(qi_fit / 1e6) * 1.2])

    ax.text(
        *text_pos,
        f"{pitch_label} µm",
        transform=ax.transAxes,
        fontsize=12,
        va=va,
        ha=ha,
        bbox=dict(facecolor="white", edgecolor="black"),
    )
    ax.set_xscale("log")
    # ax.set_yscale("log")


def fit_qi2(
    res_params,
    base_pth,
    min_photon_vec=None,
    max_photon_vec=None,
    name=None,
    bounds=([0, 0, 0, 0], [1e8, 3e7, 1e6, 5]),
    q_other_bounds=None,
    use_gamma=False,
    exclude=None,
    show_all=True,
    wide=False,
):
    """
    Fits the internal quality factor (Qi) vs. photon number for a set of resonators.

    This function processes resonator measurement data, fits the Qi data to a TLS model,
    plots the results, and returns the fitted parameters.

    Args:
        res_params (pd.DataFrame): DataFrame containing resonator parameters.
                                   Must include 'photon_number', 'frequency_Hz',
                                   'q_internal', 'q_internal_err'.
                                   Can also include 'resonator_id', 'pitch', 'target_freq', 'temp'.
        base_pth (str): Base path for saving the output plot.
        min_photon_vec (list, optional): Vector of minimum photon numbers for fitting.
        max_photon_vec (list, optional): Vector of maximum photon numbers for fitting.
        name (str, optional): Name for the output plot file.
        bounds (tuple, optional): Bounds for the curve fitting.
        use_gamma (bool, optional): If True, fits 1/Qi instead of Qi. Defaults to False.

    Returns:
        dict: A dictionary containing the fitted parameters and other metadata.
    """
    plt.rcParams["lines.markersize"] = 6
    unique_resonators = (
        res_params["resonator_id"].unique()
        if "resonator_id" in res_params.columns
        else [0]
    )
    if exclude is None:
        exclude = [None] * len(unique_resonators)
    if q_other_bounds is None:
        q_other_bounds = np.array(len(unique_resonators) * bounds[1][1])

    if len(unique_resonators) > 6:
        fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.flatten()

    results = {
        "params": [],
        "errors": [],
        "qi0": [],
        "qi_hi": [],
        "nn_min": [],
        "nn_max": [],
        "qc": [],
        "qc_err": [],
        "freqs": [],
        "pitch": [],
        "target_freq": [],
    }

    for i, res_id in enumerate(unique_resonators):
        if i >= len(axes):
            break

        res_data = (
            res_params[res_params["resonator_id"] == res_id]
            if "resonator_id" in res_params.columns
            else res_params
        )
        res_data = res_data.sort_values(by="photon_number")

        nn = res_data["photon_number"].values
        freq = res_data["frequency_Hz"].iloc[0]
        temp = (
            res_data["temp"].iloc[0] if "temp" in res_data.columns else 20e-3
        )  # Default temp

        mask = photon_mask(nn, min_photon_vec, max_photon_vec, i, exclude[i])
        nn_fit, qi_fit, qi_err = (
            nn[mask],
            res_data["q_internal"].values[mask],
            res_data["q_internal_err"].values[mask],
        )

        fit_func = (
            (
                lambda n, Qtls0, Qoth, nc, beta: Gamma_tot(
                    n, temp, freq, Qtls0, Qoth, nc, beta
                )
            )
            if use_gamma
            else (
                lambda n, Qtls0, Qoth, nc, beta: Qtotn(
                    n, temp, freq, Qtls0, Qoth, nc, beta
                )
            )
        )
        Qtls_func = lambda n, Qtls0, nc, beta: Qtls(n, temp, freq, Qtls0, nc, beta)

        y_data_fit = 1 / qi_fit if use_gamma else qi_fit
        y_sigma_fit = qi_err / qi_fit**2 if use_gamma else qi_err

        bounds[1][1] = q_other_bounds[i]
        p, err = _fit_qi_model(nn_fit, y_data_fit, y_sigma_fit, fit_func, bounds)

        results["params"].append(p)
        results["errors"].append(err)
        results["qi0"].append(1 / fit_func(0, *p) if use_gamma else fit_func(0, *p))
        results["qi_hi"].append(np.max(qi_fit))
        results["nn_min"].append(np.min(nn))
        results["nn_max"].append(np.max(nn))
        results["qc"].append(np.nanmedian(res_data["q_coupling"].values))
        results["qc_err"].append(np.nanmedian(res_data["q_coupling_err"].values))
        results["freqs"].append(np.nanmedian(res_data["frequency_Hz"].values))
        results["pitch"].append(
            res_data["pitch"].iloc[0] if "pitch" in res_data.columns else ""
        )
        results["target_freq"].append(
            res_data["target_freq"].iloc[0] if "target_freq" in res_data.columns else ""
        )

        pitch_label = res_data["pitch"].iloc[0] if "pitch" in res_data.columns else ""
        _plot_qi_fit(
            axes[i],
            nn_fit,
            qi_fit,
            qi_err,
            p,
            fit_func,
            use_gamma,
            pitch_label,
            nn,
            res_data["q_internal"].values,
            show_all=show_all,
            wide=wide,
            Qtls_func=Qtls_func,
        )

    for ax in axes:
        ax.set_xlabel(r"$\langle n \rangle$")
        ax.set_ylabel(r"$Q_i \: (10^6)$")

    fig.tight_layout()
    if name:
        try:
            fig.savefig(f"{base_pth}{name}_qi.png", dpi=300)
        except Exception as e:
            print(f"Failed to save figure: {e}")

    # Finalize results dictionary
    final_results = {
        "qtls0": np.array([p[0] for p in results["params"]]),
        "qother": np.array([p[1] for p in results["params"]]),
        "nc": np.array([p[2] for p in results["params"]]),
        "beta": np.array([p[3] for p in results["params"]]),
        "qtls0_err": np.array([e[0] for e in results["errors"]]),
        "qother_err": np.array([e[1] for e in results["errors"]]),
        "nc_err": np.array([e[2] for e in results["errors"]]),
        "beta_err": np.array([e[3] for e in results["errors"]]),
    }
    for key in [
        "qi0",
        "qi_hi",
        "nn_min",
        "nn_max",
        "qc",
        "qc_err",
        "freqs",
        "pitch",
        "target_freq",
    ]:
        final_results[key] = np.array(results[key])

    return final_results


def photon_mask(nn, min_photon_vec, max_photon_vec, i, exclude=None):
    min_val = (
        min_photon_vec[i]
        if min_photon_vec is not None and i < len(min_photon_vec)
        else None
    )
    max_val = (
        max_photon_vec[i]
        if max_photon_vec is not None and i < len(max_photon_vec)
        else None
    )

    mask = (
        (nn >= min_val) if min_val is not None else np.ones(len(nn), dtype=bool)
    ) & ((nn <= max_val) if max_val is not None else np.ones(len(nn), dtype=bool))

    if exclude is not None and len(exclude) > 0:
        exclude = np.array(exclude)
        mask[exclude] = False

    return mask


def plot_res_pars(params_list, labs, base_pth, name=None):
    plt.rcParams["lines.markersize"] = 10
    sns.set_palette(colors)
    fig, ax = plt.subplots(2, 4, figsize=(13, 6))
    ax = ax.flatten()
    i = 0
    if name is not None:
        fnames = name + "_"
    for params, l in zip(params_list, labs):
        try:
            if name is None:
                fnames += params["meas"] + "_"
        except:
            pass
        ax[0].errorbar(
            params["pitch"],
            params["qc"] / 1e6,
            yerr=params["qc_err"] / 1e6,
            fmt=".-",
            label=l,
        )
        ax[1].errorbar(
            params["pitch"],
            params["qtls0"] / 1e6,
            yerr=params["qtls0_err"] / 1e6,
            fmt=".",
        )
        ax[2].errorbar(
            params["pitch"],
            params["qother"] / 1e6,
            yerr=params["qother_err"] / 1e6,
            fmt=".",
        )
        ax[2].plot(
            params["pitch"],
            params["qi_hi"] / 1e6,
            ".",
            label="Qi high",
        )
        # ax[2].set_yscale("log")
        ax[3].errorbar(params["pitch"], params["nc"], yerr=params["nc_err"], fmt=".")
        ax[4].errorbar(
            params["pitch"], params["beta"], yerr=params["beta_err"], fmt="."
        )
        ax[5].plot(params["pitch"], params["freqs"] / params["target_freq"] / 1e9, ".")
        ax[6].plot(params["pitch"], params["qother"] / params["qtls0"], ".")
        # ax[5].plot((params['pitch'],(params['pitch']), params['qi0']/params['qc'],params['qi_hi']/params['qc']), '.-', label=l, color=colors[i])
        # ax[5].plot(params['pitch'], params['qi_hi']/params['qc'], '.', label=l, color=colors[i])

        if ax[2].get_ylim()[1] > 30:
            ax[2].set_ylim(0, np.nanmax(params["qother"] / 1e6) * 1.1)
        # ax[1].set_ylim(0,np.nanmax(params['qtls0']/1e6)*1.1)

        if ax[2].get_ylim()[1] > 13:
            ax[2].set_ylim(0, np.nanmax(params["qother"] / 1e6) * 1.2)
        # ax[3].set_ylim(0,np.nanmax(qtls0/1e6)*1.1)
        i += 1
    # ax[0].set_ylim([0, 2])
    ax[2].set_ylim(bottom=0)
    ax[0].legend()
    ax[0].set_ylabel("$Q_c \; (10^6)$")

    ax[1].set_ylabel(r"$Q_{\mathrm{TLS}} \; ( 10^6) $")
    ax[2].set_ylabel(r"$Q_{\mathrm{other}}  \; (10^6) $")
    ax[4].set_ylabel("$\\beta$")
    ax[3].set_ylabel("$n_c$")
    ax[5].set_ylabel("Frequency/Target frequency")
    ax[6].set_ylabel("$Q_{other}/Q_{TLS}$")
    for a in ax:
        a.set_xlabel("Gap width ($\mu$m)")
    fig.tight_layout()

    fig.savefig(base_pth + fnames + "params_tls_full.png", dpi=300)

    return ax

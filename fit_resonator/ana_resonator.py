import os
import re
import time
import traceback
from collections import Counter
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cs
import seaborn as sns

import meas_analysis.handy as hy
import scresonators.fit_resonator.fit as scfit
import scresonators.fit_resonator.resonator as scres
import pyCircFit_v3 as cf

colors = ["#4053d3", "#b51d14", "#ddb310", "#658b38", "#7e1e9c", "#75bbfd", "#cacaca"]


def get_resonators(folder, pth, pattern):

    # List of files
    pth = pth + folder
    file_list0 = os.listdir(pth)
    # print(len(file_list0))
    file_list = [file for file in file_list0 if re.match(pattern, file)]

    tokens = []
    # Get a list of resonators
    for i in range(len(file_list)):
        tokens.append(re.findall(pattern, file_list[i]))

    values = [int(token) for sublist in tokens for token in sublist]
    resonators = set(values)

    frequency = Counter(values)
    print(frequency)

    resonators = np.array(list(resonators))
    resonators.sort()
    return resonators, file_list


def convert_power(res_params):
    for i in range(len(res_params)):
        res_params[i]["lin_power"] = res_params[i]["pow"]
        res_params[i]["pow"] = np.log10(res_params[i]["pow"]) * 20 - 30
        # res_params[i]['freqs'] = res_params[i]['freqs']*1e6
    return res_params


def get_resonator_power_list(pattern, file_list0):
    # Grab all the files for a given resonator, then sort by power.
    file_list = [file for file in file_list0 if re.match(pattern, file)]
    tokens = []
    for i in range(len(file_list)):
        tokens.append(re.findall(pattern, file_list[i])[0])
    powers = np.array(tokens, dtype=float)
    inds = np.argsort(powers)

    file_list = [file_list[i] for i in inds]
    file_list = file_list[::-1]
    print(file_list)
    return file_list


def get_temp_list(pth_base, max_temp=1500):

    directories = [
        name
        for name in os.listdir(pth_base)
        if os.path.isdir(os.path.join(pth_base, name))
    ]
    directories = sorted(directories)
    print(directories)

    temps = np.array([float(d[7:]) for d in directories])
    print(temps)

    inds = np.where(temps < max_temp)
    temps = temps[inds]
    directories = np.array(directories)[inds]
    inds = np.argsort(temps)
    temps = temps[inds]
    directories = directories[inds]
    print(directories)
    return temps, directories


def fit_resonator(
    data,
    filename,
    output_path,
    fit_type="DCM",
    plot=False,
    pre="circle",
    fix_freq=False,
):
    # fit type DCM, CPZM
    my_resonator = scres.Resonator()
    my_resonator.outputpath = output_path
    my_resonator.filename = filename
    my_resonator.from_columns(data["freqs"], data["amps"], data["phases"])
    my_resonator.fix_freq = fix_freq
    # Set fit parameters

    MC_iteration = 4
    MC_rounds = 1e3
    MC_fix = []
    manual_init = None
    if plot:
        fmt = "png"
    else:
        fmt = None
    my_resonator.preprocess_method = pre  # Preprocess method: default = linear
    my_resonator.filepath = "./"  # Path to fit output
    # Perform a fit on the data with given parameters
    my_resonator.fit_method(
        fit_type,
        MC_iteration,
        MC_rounds=MC_rounds,
        MC_fix=MC_fix,
        manual_init=manual_init,
        MC_step_const=0.3,
    )
    output = my_resonator.fit(fmt)
    return output


# conf_array = [Q_conf, Qi_conf, Qc_conf, Qc_Re_conf, phi_conf, w1_conf]
# Q, Qc, Frequency, Phase


def check_phase(data):
    if (np.max(data) - np.min(data)) > np.pi:
        data = data * np.pi / 180
    return data


def grab_data(pth, fname, meas_type="vna", slope=0):
    data, attrs = hy.prev_data(pth, fname)
    # Reformat data for scres package
    if meas_type == "vna":
        data["phases"] = np.unwrap(data["phases"][0])
        data["phases"] = check_phase(data["phases"])
        data["freqs"] = data["fpts"][0]
        data["amps"] = data["mags"][0]
        data["phases"] = data["phases"] - slope * data["freqs"]
    elif meas_type == "soc":
        # slope = np.polyfit(data['xpts'][0], np.unwrap(data['phases'][0]), 1)
        data["phases"] = np.unwrap(data["phases"])
        if True:  # np.floor(data["xpts"][0][0] / fny) % 2 == 0:
            data["phases"] = data["phases"][0] + slope * data["xpts"][0]
        else:
            data["phases"] = data["phases"][0] - slope * data["xpts"][0]

        # if (
        #     data["phases"][0][-1] < data["phases"][0][-2] < data["phases"][0][-3]
        #     or data["phases"][0][2] < data["phases"][0][1] < data["phases"][0][0]
        # ):
        #     data["phases"] = data["phases"][0] + slope * data["xpts"][0]
        #     # data["phases"] = -data["phases"][0] - slope * data["xpts"][0]
        # else:
        #     data["phases"] = data["phases"][0] - slope * data["xpts"][0]
        data["phases"] = np.unwrap(data["phases"])
        data["amps"] = np.log10(data["amps"][0]) * 20
        data["freqs"] = data["xpts"][0] * 1e6
    return data, attrs


def plot_raw_data(data, phs_off=0, amp_off=0, circ_only=True):
    phs = data["phases"] + phs_off
    amp = 10 ** ((data["amps"] - amp_off) / 20)
    if not circ_only:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(data["freqs"] / 1e9, data["phases"] + phs_off)
        ax[1].plot(data["freqs"] / 1e9, data["amps"] - amp_off)
        ax[0].set_xlabel("Frequency (GHz)")
        ax[0].set_ylabel("Phase (rad)")
        ax[1].set_xlabel("Frequency (GHz)")
        ax[1].set_ylabel("Amplitude (dB)")

        fig.tight_layout()
    plt.figure()
    plt.plot(amp / np.max(amp) * np.cos(phs), amp / np.max(amp) * np.sin(phs), ".")
    # plt.gca().set_aspect("equal", adjustable="box")


def fit_phase(data):
    # Fit the phase to a line
    # use left 10% of data
    freq_range = np.max(data["freqs"]) - np.min(data["freqs"])
    inds = np.where(data["freqs"] > np.max(data["freqs"]) - 0.2 * freq_range)
    mf = np.mean(data["freqs"])
    slope = np.polyfit(data["freqs"][inds] - mf, data["phases"][inds], 1)
    # print(slope[0])
    data["phases"] = data["phases"] - slope[0] * (data["freqs"] - mf) - slope[1]
    data["phases"] = np.unwrap(data["phases"])

    return data


def combine_data(data1, data2, fix_freq=True, meas_type="soc"):
    data = {}
    mp = np.mean(data1["phases"])
    data1["phases"] = data1["phases"] - mp
    data2["phases"] = data2["phases"] - mp
    if fix_freq:

        phase_interp = interp1d(
            data1["freqs"], data1["phases"], fill_value="extrapolate"
        )
        ph = phase_interp(data2["freqs"][0])
        dphase = data2["phases"][0] - ph
        data2["phases"] = data2["phases"] - dphase
    keys_list = ["freqs", "amps", "phases"]
    for key in keys_list:
        data[key] = np.concatenate([data1[key], data2[key]])

    inds = data["freqs"].argsort()
    for key in keys_list:
        data[key] = data[key][inds]

    data["phases"] = np.unwrap(data["phases"])
    if meas_type == "vna":
        data["vna_power"] = data1["vna_power"]
    return data


def stow_data(params, res_params, j, power, err):
    # Put all the data for a given resonator/temp in arrays.
    power = np.array(power)
    q = np.array([params[k][0] for k in range(len(params))])
    qc = np.array([params[k][1] for k in range(len(params))])
    freq = np.array([params[k][2] for k in range(len(params))])
    phase = np.array([params[k][3] for k in range(len(params))])
    qi_phi = 1 / (1 / q - 1 / qc)
    Qc_comp = qc / np.exp(1j * phase)
    Qi = (q**-1 - np.real(Qc_comp**-1)) ** -1

    res_params[j]["pow"].append(power)
    res_params[j]["q"].append(q)
    res_params[j]["qc"].append(qc)
    res_params[j]["freqs"].append(freq)
    res_params[j]["phs"].append(phase)
    res_params[j]["qi_phi"].append(qi_phi)
    res_params[j]["qi"].append(Qi)

    res_params[j]["phs_err"].append(np.array([err[i][4] for i in range(len(err))]))
    res_params[j]["q_err"].append(np.array([err[i][0] for i in range(len(err))]))
    res_params[j]["qi_err"].append(np.array([err[i][1] for i in range(len(err))]))
    res_params[j]["qc_err"].append(np.array([err[i][2] for i in range(len(err))]))
    res_params[j]["qc_real_err"].append(np.array([err[i][3] for i in range(len(err))]))
    res_params[j]["f_err"].append(np.array([err[i][5] for i in range(len(err))]))

    fig, ax = plt.subplots(2, 1, figsize=(6, 7.5), sharex=True)
    ax[0].plot(power, Qi, ".", markersize=6)
    ax[0].set_ylabel("$Q_i$")
    ax[1].plot(power, qc, ".", markersize=6)
    ax[1].set_ylabel("$Q_c$")
    ax[1].set_xlabel("Power (dBm)")
    fig.suptitle("$f_0 = $ {:3.3f} GHz".format(freq[0] / 1e9))
    fig.tight_layout()

    return res_params


def stow_data_oth(params, res_params, j, power):
    # Put all the data for a given resonator/temp in arrays.
    power = np.array(power)
    q = np.array([params[k]["Qtot"] for k in range(len(params))])
    qc = np.array([params[k]["Qc"] for k in range(len(params))])
    Qi = np.array([params[k]["Qi"] for k in range(len(params))])
    freq = np.array([params[k]["fr"] for k in range(len(params))])
    phase = np.array([params[k]["phi"] for k in range(len(params))])
    # qi_phi = 1 / (1 / q - 1 / qc)
    # Qc_comp = qc / np.exp(1j * phase)
    # Qi = (q**-1 - np.real(Qc_comp**-1)) ** -1

    res_params[j]["pow"].append(power)
    res_params[j]["q"].append(q)
    res_params[j]["qc"].append(qc)
    res_params[j]["freqs"].append(freq)
    res_params[j]["phs"].append(phase)
    # res_params[j]["qi_phi"].append(qi_phi)
    res_params[j]["qi"].append(Qi)

    res_params[j]["phs_err"].append(
        np.array([params[i]["phi_stderr"] for i in range(len(params))])
    )
    res_params[j]["q_err"].append(
        np.array([params[i]["Qtot_stderr"] for i in range(len(params))])
    )
    res_params[j]["qi_err"].append(
        np.array([params[i]["Qi_stderr"] for i in range(len(params))])
    )
    res_params[j]["qc_err"].append(
        np.array([params[i]["Qc_stderr"] for i in range(len(params))])
    )
    res_params[j]["f_err"].append(
        np.array([params[i]["fr_stderr"] for i in range(len(params))])
    )

    fig, ax = plt.subplots(2, 1, figsize=(6, 7.5), sharex=True)
    ax[0].plot(power, Qi, ".", markersize=6)
    ax[0].set_ylabel("$Q_i$")
    ax[1].plot(power, qc, ".", markersize=6)
    ax[1].set_ylabel("$Q_c$")
    ax[1].set_xlabel("Power (dBm)")
    fig.suptitle("$f_0 = $ {:3.3f} GHz".format(freq[0] / 1e9))
    fig.tight_layout()

    return res_params


def norm_data(data):
    amp = 10 ** ((data["amps"]) / 20)
    phs = data["phases"]
    z = amp * np.exp(1j * phs)
    z = z / np.max(z)
    data["amps"] = np.log10(np.abs(z)) * 20
    data["phases"] = np.angle(z)
    data["x"] = np.real(z)
    data["y"] = np.imag(z)
    return data


def analyze_sweep_gen(
    directories,
    pth_base,
    img_pth,
    nfiles=3,
    name="res",
    plot=False,
    min_power=-120,
    meas_type="vna",
    slope=0,
    fix_freq=True,
    fitphase=False,
):
    resonators, file_list, ends = resonator_list(
        directories, pth_base, nfiles, meas_type
    )

    # Initialize dict by getting list of resonators, creating dict with len = n resonators
    res_params = [None] * len(resonators)
    for i in range(len(resonators)):
        res_params[i] = {
            "freqs": [],
            "phs": [],
            "q": [],
            "qi": [],
            "qc": [],
            "qi_phi": [],
            "pow": [],
            "qi_err": [],
            "q_err": [],
            "phs_err": [],
            "qc_err": [],
            "qc_real_err": [],
            "f_err": [],
        }

    # Each directory is a temperature
    for i in range(len(directories)):
        start = time.time()
        output_path = img_pth + name + "_" + directories[i] + "/"
        pth = pth_base + directories[i]
        for j in range(len(resonators)):
            params, err, power = [], [], []
            # Grab all the files for a given resonator, then sort by power.
            for k in range(len(file_list[i][j])):  # power
                fname = file_list[i][j][k]
                try:
                    data, attrs = load_resonator(
                        fname, pth, nfiles, slope, meas_type, ends, fix_freq=True
                    )
                    if fitphase:
                        data = fit_phase(data)
                    # data = norm_data(data)
                except:
                    # traceback.print_exc()
                    continue

                # Skip really noisy ones since they are slow
                if meas_type == "vna":
                    if data["vna_power"][0] < min_power:
                        continue
                    power.append(data["vna_power"][0])
                    print(power[-1])
                else:
                    if attrs["gain"] < min_power:
                        continue

                    power.append(np.log10(attrs["gain"]) * 20 - 30)
                    # power.append(attrs["gain"])

                    print(power[-1])

                try:
                    output = fit_resonator(
                        data, file_list[i][j][k], output_path, plot=plot, fix_freq=True
                    )
                    params.append(output[0])
                    err.append(output[1])
                except Exception as error:
                    print("An exception occurred:", error)
                    params.append(np.nan * np.ones(4))
                    err.append(np.nan * np.ones(6))
            res_params = stow_data(params, res_params, j, power, err)

            print("Time elapsed: ", time.time() - start)

    for i in range(len(resonators)):
        for key in res_params[i].keys():
            res_params[i][key] = np.array(res_params[i][key])

    return res_params


def plot_power(res_params, cfg, base_pth, use_pitch=True):
    sns.set_palette("coolwarm", len(res_params))
    plt.rcParams["lines.markersize"] = 4
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    fig2, ax2 = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    for i in range(len(res_params)):
        if use_pitch:
            l = cfg["pitch"][i]
        else:
            l = round(np.min(res_params[i]["freqs"] / 1e9), 4)
        inds = np.where(
            (res_params[i]["pow"][0] >= cfg["min_power"])
            & (res_params[i]["pow"][0] <= cfg["max_power"])
        )
        ax[0].semilogy(
            res_params[i]["pow"][0][inds], res_params[i]["qi"][0][inds], ".-", label=l
        )
        ax[1].semilogy(
            res_params[i]["pow"][0][inds],
            res_params[i]["qi"][0][inds] / np.nanmax(res_params[i]["qi"][0]),
            ".-",
            label=l,
        )
        ax2[0].plot(
            res_params[i]["pow"][0][inds],
            1e6
            * (
                res_params[i]["freqs"][0][inds]
                / np.nanmin(res_params[i]["freqs"][0][inds])
                - 1
            ),
            ".-",
            label=l,
        )
        ax2[1].plot(
            res_params[i]["pow"][0][inds],
            res_params[i]["qc"][0][inds] / np.nanmin(res_params[i]["qc"][0][inds]),
            ".-",
            label=l,
        )

    ax[1].set_xlabel("Power")
    ax[0].set_ylabel("$Q_i$")
    ax[1].set_ylabel("$Q_i/Q_{i,max}$")
    ax[1].legend(title="Gap", fontsize=8)

    ax2[1].set_xlabel("Power")
    ax2[0].set_ylabel("$\Delta f/f$ (ppm)")
    ax2[1].set_ylabel("$Q_c/Q_{c,min}$")
    fig.tight_layout()
    fig2.tight_layout()
    try:
        fig2.savefig(base_pth + cfg["res_name"] + "_Qcfreq_pow.png", dpi=300)
        plt.show()

        fig.savefig(base_pth + cfg["res_name"] + "_Qi_pow.png", dpi=300)
        plt.show()
    except:
        print("Error in plotting")


def plot_temp(res_params, cfg, use_pitch, base_pth, xval="temp"):

    plt.rcParams["lines.markersize"] = 6
    plt.rcParams["lines.linewidth"] = 1.5
    inds = np.argsort(cfg["temps"])
    en = 1e-3 * cfg["temps"][inds] * cs.k / cs.h / res_params[i]["freqs"][inds, 0]
    if xval == "temp":
        x = cfg["temps"][inds]
        xlab = "Temperature (mK)"
    else:
        x = en
        xlab = "$k_B T / h f_0$"
    if use_pitch:
        l = cfg["pitch"][i]
    else:
        l = round(np.min(res_params[i]["freqs"] / 1e9), 4)
    j = 0
    # Temperature sweep
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    for i in range(len(res_params)):
        inds2 = (
            res_params[i]["qi"][inds, j] / np.max(res_params[i]["qi"][inds, j]) > 0.72
        )
        min_freq = np.nanmin(res_params[i]["freqs"][inds, :])
        x = cfg["temps"][inds2]

        ax[0].plot(
            x,
            res_params[i]["qi"][inds2, j] / np.max(res_params[i]["qi"][inds2, j]),
            ".-",
        )
        ax[1].plot(x, (res_params[i]["freqs"][inds2, j] - min_freq) / min_freq, ".-")

    ax[0].set_ylabel("$Q_i/Q_{i,max}$")
    ax[1].set_xlabel(xlab)
    ax[1].set_ylabel("$\Delta f/f_0$")
    fig.tight_layout()
    plt.savefig(base_pth + "_" + cfg["res_name"] + "_temp_sweep.png", dpi=300)


def plot_power_temp(res_params, i, cfg, base_pth, use_cbar=False, xval="temp"):

    plt.rcParams["lines.markersize"] = 4
    plt.rcParams["lines.linewidth"] = 1

    inds = np.argsort(cfg["temps"])
    en = 1e-3 * cfg["temps"][inds] * cs.k / cs.h / res_params[i]["freqs"][inds, 0]
    if xval == "temp":
        x = cfg["temps"][inds]
        xlab = "Temperature (mK)"
    else:
        x = en
        xlab = "$k_B T / h f_0$"

    min_freq = np.nanmin(res_params[i]["freqs"][inds, :])
    sns.set_palette("coolwarm", n_colors=res_params[0]["pow"].shape[1])

    # Temperature sweep
    fig, ax = plt.subplots(4, 1, figsize=(6, 9), sharex=True)
    for j in range(res_params[i]["pow"].shape[1]):
        ax[0].plot(x, res_params[i]["qi"][inds, j], ".-")
        ax[1].plot(
            x, res_params[i]["qi"][inds, j] / np.max(res_params[i]["qi"][inds, j]), ".-"
        )
        ax[2].plot(x, res_params[i]["qc"][inds, j], ".-")
        ax[3].plot(x, (res_params[i]["freqs"][inds, j] - min_freq) / min_freq, ".-")

    if use_cbar:
        norm = plt.Normalize(np.min(cfg["temps"]), np.max(cfg["temps"]))
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        ax[1].figure.colorbar(sm, ax=ax[1])

        norm = plt.Normalize(np.min(res_params[i]["pow"]), np.max(res_params[i]["pow"]))
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        ax[0].figure.colorbar(sm, ax=ax[0])

    ax[0].set_ylabel("$Q_i$")
    ax[1].set_ylabel("$Q_i/Q_i(0)$")
    ax[2].set_ylabel("$Q_c$")
    ax[3].set_xlabel("Temperature (mK)")
    ax[3].set_xlabel(xlab)
    ax[3].set_ylabel("$\Delta f/f_0$")

    fig.tight_layout()
    fig.savefig(base_pth + cfg["res_name"] + "_Qi_temp_" + str(i) + ".png", dpi=300)

    sns.set_palette("coolwarm", n_colors=res_params[0]["pow"].shape[0])
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    # Plot power dependence
    for j in inds:
        ax[0].plot(
            res_params[i]["pow"][j, :],
            res_params[i]["qi"][j, :],
            ".-",
            label=int(cfg["temps"][j]),
        )
        ax[1].plot(
            res_params[i]["pow"][j, :],
            res_params[i]["qi"][j, :] / np.max(res_params[i]["qi"][j, :]),
            ".-",
        )

    ax[0].set_ylabel("$Q_i$")
    ax[1].set_ylabel("$Q_i/Q_i(0)$")
    ax[1].set_xlabel("Power (dBm)")

    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(base_pth + cfg["res_name"] + "_Qi_pow_" + str(i) + ".png", dpi=300)


def plot_res_pars(params_list, labs, base_pth, name=None):
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True)
    ax = ax.flatten()
    fnames = ""
    sns.set_palette(colors)
    ax[0].set_ylabel("Frequency (GHz)")
    ax[1].set_ylabel("Frequency/Designed Freq")
    ax[2].set_ylabel("Phase (rad)")

    if name is not None:
        fnames = name + "_"
    for params, l in zip(params_list, labs):
        try:
            if name is None:
                fnames += params["meas"] + "_"
        except:
            pass

        ax[0].plot(params["pitch"], params["freqs"] / 1e9, ".", label=l)
        ax[1].plot(
            params["pitch"],
            params["freqs"] / 1e9 / params["target_freq"],
            ".-",
            label=l,
        )
        ax[2].plot(params["pitch"], params["phs"], ".-", label=l)

    for a in ax:
        a.set_xlabel("Gap width ($\mu$m)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(base_pth + fnames + "params_res_full.png", dpi=300)


def reorder(params, res_params, use_pitch=True):

    params["pitch"] = params["pitch"][0 : len(res_params)]
    if use_pitch:
        ord = np.argsort(params["pitch"])
        res_params = [res_params[i] for i in ord]
        params["pitch"] = [params["pitch"][i] for i in ord]
        params["target_freq"] = [params["target_freq"][i] for i in ord]
    return params, res_params


def resonator_list(directories, pth_base, nfiles, meas_type):
    if nfiles == 3:
        if meas_type == "vna":
            pattern_end = "dbm_"
            ends = ["wide1", "narrow", "wide2"]
        else:
            pattern_end = "_"
            ends = ["wideleft", "narrow", "wideright"]
        pattern0 = r"res_(\d+)_\d{2,5}" + pattern_end + ends[0]
    elif nfiles == 2:
        pattern0 = r"res_(\d+)_\d{2,5}dbm_wide"
        ends = []
    else:
        if meas_type == "vna":
            pattern_end = "dbm"
        else:
            pattern_end = ""
        pattern0 = r"res_(\d+)_\d{2,5}" + pattern_end
        ends = [""]

    resonators, _ = get_resonators(directories[0], pth_base, pattern0)
    file_list_full = []
    for i in range(len(directories)):
        file_list_full.append([])
        # Grab all the files for a given resonator, then sort by power.
        resonators, file_list0 = get_resonators(directories[i], pth_base, pattern0)

        for j in range(len(resonators)):
            if nfiles == 3:
                pattern = (
                    "res_{:d}_".format(resonators[j])
                    + "(\d{2,5})"
                    + pattern_end
                    + ends[0]
                )
            elif nfiles == 2:
                pattern = "res_{:d}_".format(resonators[j]) + "(\d{2,3})dbm_wide"
            else:
                pattern = (
                    "res_{:d}_".format(resonators[j])
                    + "(\d{2,5})"
                    + pattern_end
                    + ends[0]
                )
            file_list = get_resonator_power_list(pattern, file_list0)
            file_list_full[i].append(file_list)

    return resonators, file_list_full, ends
    # Initialize dict by getting list of resonators, creating dict with len = n resonators


def load_resonator(fname, pth, nfiles, slope, meas_type, ends, fix_freq):
    # file_list[k]

    data1, attrs = grab_data(pth, fname, meas_type, slope)
    if nfiles == 1:
        data = data1
        data["phases"] = data["phases"] - np.mean(data["phases"])
    if nfiles > 1:
        file2 = fname.replace(ends[0], ends[1])
        data2, _ = grab_data(pth, file2, meas_type, slope)
        data = combine_data(data1, data2, fix_freq, meas_type)
    if nfiles > 2:
        file3 = fname.replace(ends[0], ends[2])
        data3, _ = grab_data(pth, file3, meas_type, slope)
        data = combine_data(data, data3, fix_freq, meas_type)

    return data, attrs


def plot_all(
    directories,
    pth_base,
    output_path,
    name="res",
    min_power=-120,
    max_power=-25,
    norm=False,
    nfiles=3,
    meas_type="vna",
    slope=0,
    circ=False,
    half_norm=False,
):

    resonators, file_list, ends = resonator_list(
        directories, pth_base, nfiles, meas_type
    )
    nres = len(resonators)

    sns.set_palette("coolwarm", n_colors=int(len(file_list[0][0])))
    fig, ax = plt.subplots(2, 4, figsize=(10, 7))
    fig2, ax2 = plt.subplots(2, 4, figsize=(10, 7))
    ax = ax.flatten()
    ax2 = ax2.flatten()
    output = []
    # Each directory is a temperature
    for i in range(len(directories)):
        pth = pth_base + directories[i]
        for j in range(len(resonators)):
            # for k in range(1):
            for k in range(len(file_list[i][j])):  # power
                fname = file_list[i][j][k]
                try:
                    data, _ = load_resonator(
                        fname, pth, nfiles, slope, meas_type, ends, fix_freq=True
                    )
                    data = fit_phase(data)
                except:
                    traceback.print_exc()
                    continue
                if meas_type == "vna":
                    if (
                        data["vna_power"][0] > max_power
                        or data["vna_power"][0] < min_power
                    ):
                        continue

                if norm:
                    x = 10 ** (data["amps"] / 20) * np.cos(data["phases"])
                    y = 10 ** (data["amps"] / 20) * np.sin(data["phases"])
                    z = scfit.preprocess_circle(
                        data["freqs"],
                        x + 1j * y,
                        output_path="hi",
                        fix_freq=True,
                        plot_extra=True,
                    )
                    ax[j].plot(
                        (data["freqs"] - np.mean(data["freqs"])) / 1e3,
                        np.log10(np.abs(z)) * 20,
                        linewidth=1,
                    )
                    ax2[j].plot(data["freqs"] / 1e9, np.angle(z), linewidth=1)
                    if circ:
                        plt.figure()
                        plt.plot(np.real(z), np.imag(z), ".")
                elif half_norm:
                    ax[j].plot(
                        (data["freqs"] - np.mean(data["freqs"])) / 1e3,
                        data["amps"] / np.max(data["amps"]),
                        linewidth=1,
                    )
                    ax2[j].plot(
                        (data["freqs"] - np.mean(data["freqs"])) / 1e3,
                        data["phases"],
                        linewidth=1,
                    )
                    data = norm_data(data)
                    if circ:
                        plt.figure()

                        plot_raw_data(data, phs_off=0, amp_off=0, circ_only=True)

                else:
                    ax[j].plot(
                        (data["freqs"] - np.mean(data["freqs"])) / 1e3,
                        data["amps"],
                        linewidth=1,
                    )
                    ax2[j].plot(
                        (data["freqs"] - np.mean(data["freqs"])) / 1e3,
                        data["phases"],
                        linewidth=1,
                    )
                    if circ:
                        plot_raw_data(data, phs_off=0, amp_off=0, circ_only=True)

            fig.tight_layout()
            fig2.tight_layout()
            fig2.savefig(
                output_path + name + "_" + directories[i] + "_all_data_phase.png",
                dpi=300,
            )
            fig.savefig(
                output_path + name + "_" + directories[i] + "_all_data_amp.png", dpi=300
            )
    return file_list


def analyze_sweep_other(
    directories,
    pth_base,
    output_path,
    name="res",
    min_power=-120,
    max_power=-25,
    norm=False,
    nfiles=3,
    meas_type="vna",
    slope=0,
    circ=False,
    half_norm=False,
):

    resonators, file_list, ends = resonator_list(
        directories, pth_base, nfiles, meas_type
    )
    nres = len(resonators)
    res_params = [None] * len(resonators)
    for i in range(len(resonators)):
        res_params[i] = {
            "freqs": [],
            "phs": [],
            "q": [],
            "qi": [],
            "qc": [],
            "qi_phi": [],
            "pow": [],
            "qi_err": [],
            "q_err": [],
            "phs_err": [],
            "qc_err": [],
            "qc_real_err": [],
            "f_err": [],
        }
    sns.set_palette("coolwarm", n_colors=int(len(file_list[0][0])))
    fig, ax = plt.subplots(2, 4, figsize=(10, 7))
    fig2, ax2 = plt.subplots(2, 4, figsize=(10, 7))
    ax = ax.flatten()
    ax2 = ax2.flatten()
    # Each directory is a temperature
    for i in range(len(directories)):
        pth = pth_base + directories[i]
        for j in range(len(resonators)):
            output, power = [], []

            # for k in range(1):
            for k in range(len(file_list[i][j])):  # power
                fname = file_list[i][j][k]
                try:
                    data, attrs = load_resonator(
                        fname, pth, nfiles, slope, meas_type, ends, fix_freq=True
                    )
                    # data = fit_phase(data)
                except:
                    traceback.print_exc()
                    continue
                if meas_type == "vna":
                    if data["vna_power"][0] < min_power:
                        continue
                    power.append(data["vna_power"][0])
                    print(power[-1])
                else:
                    if attrs["gain"] < min_power:
                        continue

                    power.append(np.log10(attrs["gain"]) * 20 - 30)

                    print(power[-1])
                data = norm_data(data)
                output.append(
                    cf.circlefit(
                        data["freqs"], data["x"], data["y"], print_results=False
                    )
                )
            res_params = stow_data_oth(output, res_params, j, power)
            fig.tight_layout()
            fig2.tight_layout()
            fig2.savefig(
                output_path + name + "_" + directories[i] + "_all_data_phase.png",
                dpi=300,
            )
            fig.savefig(
                output_path + name + "_" + directories[i] + "_all_data_amp.png", dpi=300
            )

    for i in range(len(resonators)):
        for key in res_params[i].keys():
            res_params[i][key] = np.array(res_params[i][key])
    return file_list, res_params

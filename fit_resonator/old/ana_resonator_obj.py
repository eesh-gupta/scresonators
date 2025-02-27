import os
import re
import time
import traceback
from collections import Counter
from slab import AttrDict

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cs
import seaborn as sns

import meas_analysis.handy as hy
import scresonators.fit_resonator.fit as scfit
import scresonators.fit_resonator.resonator as scres

colors = ["#4053d3", "#b51d14", "#ddb310", "#658b38", "#7e1e9c", "#75bbfd", "#cacaca"]


class Resonator:
    def __init__(self, pth, type, file, slope, nfiles, fix_freq=True):
        self.pth = pth
        self.type = type
        self.file = file
        self.slope = slope
        self.nfiles = nfiles
        self.fix_freq = fix_freq

    def res_files(self, pattern, ends):
        try:
            data1, self.attrs = self.grab_data(self)
            if self.nfiles == 1:
                data = data1
            if self.nfiles > 1:
                file2 = self.file.replace(ends[0], ends[1])
                data2, _ = self.grab_data(self.pth, file2, type, self.slope)
                data = self.combine_data(data1, data2, self.fix_freq)
            if self.nfiles > 2:
                file3 = self.file.replace(ends[0], ends[2])
                data3, _ = self.grab_data(self.pth, file3, type, self.slope)
                data = self.combine_data(data, data3, self.fix_freq)
        except:
            traceback.print_exc()

    def fit_resonator(
        self,
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

    def check_phase(self):
        if (np.max(self.data) - np.min(self.data)) > np.pi:
            self.data = self.data * np.pi / 180

    def grab_data(self, pth, fname, meas_type="vna", slope=0):
        self.data, self.attrs = hy.prev_data(pth, fname)

        # Reformat data for scres package
        if meas_type == "vna":
            self.data["phases"] = np.unwrap(self.data["phases"][0])
            self.data["phases"] = check_phase(self.data["phases"])
            self.data["freqs"] = self.data["fpts"][0]
            self.data["amps"] = self.data["mags"][0]
        elif meas_type == "soc":
            # slope = np.polyfit(data['xpts'][0], np.unwrap(data['phases'][0]), 1)
            self.data["phases"] = np.unwrap(self.data["phases"])
            if (
                self.data["phases"][0][-1]
                < self.data["phases"][0][-2]
                < self.data["phases"][0][-3]
                or self.data["phases"][0][2]
                < self.data["phases"][0][1]
                < self.data["phases"][0][0]
            ):
                self.data["phases"] = (
                    -self.data["phases"][0] - slope * self.data["xpts"][0]
                )
            else:
                self.data["phases"] = (
                    self.data["phases"][0] - slope * self.data["xpts"][0]
                )
            self.data["phases"] = np.unwrap(self.data["phases"])
            self.data["amps"] = np.log10(self.data["amps"][0]) * 20
            self.data["freqs"] = self.data["xpts"][0] * 1e6

    def plot_raw_data(self, data, phs_off=0, amp_off=0):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(self.data["freqs"] / 1e9, self.data["phases"] + phs_off)
        ax[1].plot(self.data["freqs"] / 1e9, self.data["amps"] - amp_off)
        ax[0].set_xlabel("Frequency (GHz)")
        ax[0].set_ylabel("Phase (rad)")
        ax[1].set_xlabel("Frequency (GHz)")
        ax[1].set_ylabel("Amplitude (dB)")
        phs = self.data["phases"] + phs_off
        amp = 10 ** ((self.data["amps"] - amp_off) / 20)
        fig.tight_layout()
        plt.figure()

        plt.plot(amp * np.cos(phs), amp * np.sin(phs), ".")

    def fit_phase(self):
        # Fit the phase to a line
        # use left 10% of data
        freq_range = np.max(self.data["freqs"]) - np.min(self.data["freqs"])
        inds = np.where(
            self.data["freqs"] < np.min(self.data["freqs"]) + 0.2 * freq_range
        )
        mf = np.mean(self.data["freqs"])
        slope = np.polyfit(self.data["freqs"][inds] - mf, self.data["phases"][inds], 1)
        print(slope[0])
        self.data["phases"] = (
            self.data["phases"] - slope[0] * (self.data["freqs"] - mf) - slope[1]
        )
        self.data["phases"] = np.unwrap(self.data["phases"])

    def combine_data(data1, data2, fix_freq=True):
        data = {}
        if fix_freq:
            dphase = data2["phases"][0] - data1["phases"][-1]
            data2["phases"] = data2["phases"] - dphase
        keys_list = ["freqs", "amps", "phases"]
        for key in keys_list:
            data[key] = np.concatenate([data1[key], data2[key]])

        inds = data["freqs"].argsort()
        for key in keys_list:
            data[key] = data[key][inds]

        data["phases"] = np.unwrap(data["phases"])
        return data


class ResonatorList:
    def __init__(self, nfiles):
        self.cfg = AttrDict({})
        self.nfiles = nfiles

    def stow_data(self, params, power, err):
        self.cfg.lin_power = []
        self.cfg.pow = []
        self.cfg.q = []
        self.cfg.qi = []
        self.cfg.qc = []
        self.cfg.qi_phi = []
        self.cfg.freqs = []
        self.cfg.phs = []
        self.cfg.qi_err = []
        self.cfg.q_err = []
        self.cfg.phs_err = []
        self.cfg.qc_err = []
        self.cfg.qc_real_err = []
        self.cfg.f_err = []

        # Put all the data for a given resonator/temp in arrays.
        power = np.array(power)
        q = np.array([params[k][0] for k in range(len(params))])
        qc = np.array([params[k][1] for k in range(len(params))])
        freq = np.array([params[k][2] for k in range(len(params))])
        phase = np.array([params[k][3] for k in range(len(params))])
        qi_phi = 1 / (1 / q - 1 / qc)
        Qc_comp = qc / np.exp(1j * phase)
        Qi = (q**-1 - np.real(Qc_comp**-1)) ** -1

        self.cfg.pow.append(power)
        self.cfg.q.append(q)
        self.cfg.qc.append(qc)
        self.cfg.freqs.append(freq)
        self.cfg.phs.append(phase)
        self.cfg.qi_phi.append(qi_phi)
        self.cfg.qi.append(Qi)

        # conf_array = [Q_conf, Qi_conf, Qc_conf, Qc_Re_conf, phi_conf, w1_conf]
        # Q, Qc, Frequency, Phase

        self.cfg.phs_err.append(np.array([err[i][4] for i in range(len(err))]))
        self.cfg.q_err.append(np.array([err[i][0] for i in range(len(err))]))
        self.cfg.qi_err.append(np.array([err[i][1] for i in range(len(err))]))
        self.cfg.qc_err.append(np.array([err[i][2] for i in range(len(err))]))
        self.cfg.qc_real_err.append(np.array([err[i][3] for i in range(len(err))]))
        self.cfg.f_err.append(np.array([err[i][5] for i in range(len(err))]))

        fig, ax = plt.subplots(2, 1, figsize=(6, 7.5), sharex=True)
        ax[0].plot(power, Qi, ".", markersize=6)
        ax[0].set_ylabel("$Q_i$")
        ax[1].plot(power, qc, ".", markersize=6)
        ax[1].set_ylabel("$Q_c$")
        ax[1].set_xlabel("Power (dBm)")
        fig.suptitle("$f_0 = $ {:3.3f} GHz".format(freq[0] / 1e9))
        fig.tight_layout()

    def convert_power(self):
        self.cfg.lin_power = self.cfg.pow
        self.cfg.pow = np.log10(self.cfg.pow) * 20 - 30
        # res_params[i]['freqs'] = res_params[i]['freqs']*1e6

    def reorder(self, params, use_pitch=True):

        params["pitch"] = params["pitch"][0 : len(self.cfg)]
        if use_pitch:
            ord = np.argsort(params["pitch"])
            self.cfg = [self.cfg[i] for i in ord]
            self.cfg.pitch = [params["pitch"][i] for i in ord]
            self.cfg.target_freq = [params["target_freq"][i] for i in ord]


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


def analyze_sweep_gen(
    directories,
    pth_base,
    img_pth,
    nfiles=3,
    name="res",
    plot=False,
    min_power=-120,
    type="vna",
    slope=0,
    fix_freq=False,
):
    if nfiles == 3:
        if type == "vna":
            pattern_end = "dbm_"
            ends = ["wide1", "narrow", "wide2"]
        else:
            pattern_end = "_"
            ends = ["wideleft", "narrow", "wideright"]
    else:
        ends = [""]
        if type == "vna":
            pattern_end = "dbm"
        else:
            pattern_end = ""
    # frequency / power
    pattern0 = r"res_(\d+)_\d{1,5}" + pattern_end + ends[0]

    # Initialize dict by getting list of resonators, creating dict with len = n resonators
    resonators, file_list = get_resonators(directories[0], pth_base, pattern0)
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
        resonators, file_list0 = get_resonators(directories[i], pth_base, pattern0)
        pth = pth_base + directories[i]
        for j in range(len(resonators)):
            params, err, power = [], [], []
            # Grab all the files for a given resonator, then sort by power.
            pattern = (
                "res_{:d}_".format(resonators[j]) + "(\d{1,5})" + pattern_end + ends[0]
            )
            file_list = get_resonator_power_list(pattern, file_list0)

            for k in range(len(file_list)):
                try:
                    data1, attrs = grab_data(pth, file_list[k], type, slope)
                    if nfiles == 1:
                        data = data1
                    if nfiles > 1:
                        file2 = file_list[k].replace(ends[0], ends[1])
                        data2, _ = grab_data(pth, file2, type, slope)
                        data = combine_data(data1, data2, fix_freq)
                    if nfiles > 2:
                        file3 = file_list[k].replace(ends[0], ends[2])
                        data3, _ = grab_data(pth, file3, type, slope)
                        data = combine_data(data, data3, fix_freq)
                except:
                    traceback.print_exc()

                    continue

                # Skip really noisy ones since they are slow
                if type == "vna":
                    if data1["vna_power"][0] < min_power:
                        continue
                    power.append(data1["vna_power"][0])
                    print(power[-1])
                else:
                    if attrs["gain"] < min_power:
                        continue

                    power.append(np.log10(attrs["gain"]) * 20 - 30)

                    print(power[-1])

                try:
                    output = fit_resonator(
                        data, file_list[k], output_path, plot=plot, fix_freq=True
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


def plot_res_pars(params_list, labs, base_pth):
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True)
    ax = ax.flatten()
    fnames = ""
    sns.set_palette(colors)
    ax[0].set_ylabel("Frequency (GHz)")
    ax[1].set_ylabel("Frequency/Designed Freq")
    ax[2].set_ylabel("Phase (rad)")

    for params, l in zip(params_list, labs):
        try:
            fnames += params["res_name"] + "_"
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
        pattern0 = r"res_(\d+)_\d{2,3}dbm_wide"
    else:
        pattern0 = r"res_(\d+)_\d{2,3}dbm"

    # Initialize dict by getting list of resonators, creating dict with len = n resonators
    resonators, file_list = get_resonators(directories[0], pth_base, pattern0)
    return resonators, file_list


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
):

    resonators, file_list = resonator_list(directories, pth_base, nfiles, meas_type)
    nres = len(resonators)

    sns.set_palette("coolwarm", n_colors=int(np.ceil(len(file_list) / nres)))
    fig, ax = plt.subplots(2, 4, figsize=(10, 7))
    fig2, ax2 = plt.subplots(2, 4, figsize=(10, 7))
    ax = ax.flatten()
    ax2 = ax2.flatten()

    # Each directory is a temperature
    for i in range(len(directories)):

        resonators, file_list0 = get_resonators(directories[i], pth_base, pattern0)
        pth = pth_base + directories[i]
        for j in range(len(resonators)):
            # Grab all the files for a given resonator, then sort by power.
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
                pattern = "res_{:d}_".format(resonators[j]) + "(\d{2,3})dbm"
            file_list = get_resonator_power_list(pattern, file_list0)

            for k in range(len(file_list)):
                try:
                    if nfiles == 3:
                        data1, _ = grab_data(pth, file_list[k], meas_type, slope)
                        file2 = file_list[k].replace(ends[0], ends[1])
                        data2, _ = grab_data(pth, file2, meas_type, slope)
                        data = combine_data(data1, data2)
                        file3 = file_list[k].replace(ends[0], ends[2])
                        data3, _ = grab_data(pth, file3, meas_type, slope)
                        data = combine_data(data, data3, fix_freq=True)
                    elif nfiles == 2:
                        data1, _ = grab_data(pth, file_list[k])
                        file2 = file_list[k].replace("wide", "narrow")
                        data2, _ = grab_data(pth, file2)
                        data = combine_data(data1, data2)
                    else:
                        data1, _ = grab_data(pth, file_list[k])
                        data = data1
                    data = fit_phase(data)

                except:
                    continue
                if meas_type == "vna":
                    if (
                        data1["vna_power"][0] > max_power
                        or data1["vna_power"][0] < min_power
                    ):
                        continue

                if norm:
                    x = 10 ** (data["amps"] / 20) * np.cos(data["phases"])
                    y = 10 ** (data["amps"] / 20) * np.sin(data["phases"])
                    # z, slopes, intercept, slope2, intercept2 = scfit.preprocess_linear(x, y, normalize=10, output_path='hi', plot_extra=False)
                    z = scfit.preprocess_circle(
                        data["freqs"], x + 1j * y, output_path="hi", fix_freq=True
                    )
                    ax[j].plot(
                        (data["freqs"] - np.mean(data["freqs"])) / 1e3,
                        np.log10(np.abs(z)) * 20,
                        linewidth=1,
                    )
                    ax2[j].plot(data["freqs"] / 1e9, np.angle(z), linewidth=1)
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

            fig.tight_layout()
            fig2.tight_layout()
            fig2.savefig(
                output_path + name + "_" + directories[i] + "_all_data_phase.png",
                dpi=300,
            )
            fig.savefig(
                output_path + name + "_" + directories[i] + "_all_data_amp.png", dpi=300
            )

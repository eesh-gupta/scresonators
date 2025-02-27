from ZNB import ZNB20
from VNA_funcs import *
from slab.datamanagement import SlabFile


def do_vna_scan(VNA, file_name, expt_path, cfg, spar, att=0, plot=True):

    freq_center = cfg["freq_center"]
    freq_span = cfg["span"]
    freq_start = freq_center - 0.5 * freq_span
    freq_stop = freq_center + 0.5 * freq_span

    # bandwidth  = 200 #Hz
    power = cfg["power"]  # -50 #dBm
    scattering_parameter = (spar,)  # needs to be tuple
    trace_name = ("trace1",)  # needs to be tuple

    # VNA setup
    freq_sweep = np.linspace(freq_start, freq_stop, cfg["nb_points"])
    VNA.initialize_one_tone_spectroscopy(trace_name, scattering_parameter)
    VNA.set_startfrequency(freq_start)
    VNA.set_stopfrequency(freq_stop)
    VNA.set_points(cfg["nb_points"])
    VNA.set_power(cfg["power"])
    VNA.set_measBW(cfg["bandwidth"])
    VNA.set_sweeps(cfg["averages"])
    VNA.set_averages(cfg["averages"])
    VNA.set_averagestatus(status="on")

    ## Sanity check power setting
    power_at_device = cfg["power"] - att
    # print(VNA.get_power)

    VNA.measure()
    [amps, phases] = VNA.get_traces(("trace1",))[0]
    data = {
        "series": "",  # datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
        "amps": amps,
        "phases": phases,
        "freqs": freq_sweep,
        "vna_power": power,
        "power_at_device": power - att,
        "bandwidth": cfg["bandwidth"],
        "averages": cfg["averages"],
        "nb_points": cfg["nb_points"],
    }

    """Saving Data"""

    with SlabFile(expt_path + "//" + file_name, "w") as f:
        f.append_line("fpts", freq_sweep)
        f.append_line("mags", amps)
        f.append_line("phases", phases)
        f.append_pt("vna_power", power)
        f.append_pt("averages", cfg["averages"])
        f.append_pt("bandwidth", cfg["bandwidth"])
        f.append_pt("nb_points", cfg["nb_points"])

    #         fp1, fn1 = write_file(data, expt_path, filename=file_name)
    if plot:
        plot_all(data, filepath=expt_path)


def plot_scan(freq, amps, phase, pars=None, pinit=None, power=None, slope=None):

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(freq, amps, "k.", markersize=3)
    ax[0].set_xlabel("Frequency (MHz)")
    ax[0].set_ylabel("Amplitude")
    if pars is not None:
        q = 1 / (1 / pars[1] + 1 / pars[2]) * 1e4
        qi = pars[1] * 1e4
        lab = f"$Q$={q:.3g},\n $Q_i$={qi:.3g}"
        ax[0].plot(freq, fitter.hangerS21func_sloped(freq, *pars), label=lab)
        ax[0].plot(freq, fitter.hangerS21func_sloped(freq, *pinit))
    ax[0].set_title(f"Power: {power:.1f} dB")
    if slope is None:
        phase = np.unwrap(phase)
        slope, of = np.polyfit(freq, phase, 1)
        phase_sub = phase - slope * freq - of
    else:
        phase_sub = phase - slope * freq
    ax[1].plot(freq, np.unwrap(phase_sub), "k.-", markersize=3)
    ax[1].set_xlabel("Frequency (MHz)")
    ax[1].set_ylabel("Phase")
    ax[0].legend()
    ax2 = ax[0].twinx()
    if pars is not None:
        ax2.set_ylim(ax[0].get_ylim()[0] / pars[4], ax[0].get_ylim()[1] / pars[4])
    ax2.set_ylabel("Normalized Amplitude")

    ax[2].plot(amps * np.cos(phase_sub), amps * np.sin(phase_sub), "k.-")
    fig.tight_layout()


def power_sweep(config):

    expt_path = os.path.join(config["base_path"], config["folder"])
    try:
        os.makedirs(expt_path)
    except:
        pass

    freqs0 = copy.deepcopy(config["freqs"])
    freqs = copy.deepcopy(config["freqs"])

    pows = np.arange(0, config["nvals"]) * config["pow_inc"] + config["pow_start"]
    spans = config["span_inc"] * config["kappa_start"] * np.ones(len(freqs0))
    new_avgs = np.ones(len(freqs0))
    min_avg = 100
    nph_list = [[] for i in range(len(freqs0))]
    pars_list = [[] for i in range(len(freqs0))]
    avg_list = [[] for i in range(len(freqs0))]
    qi = np.full((len(freqs), len(gains)), np.nan)
    pow_list = []

    qick_config = {
        "pow_inc": config["pow_inc"],
        "gain": 1,
        "bw": config["bw"],
        "final_delay": config["final_delay"],
        "soft_avgs": config["soft_avgs"],
        "max_reps": config["max_reps"],
        "trig_time": config["trig_time"],
    }

    # for j in range(len(freqs)):
    for i, pow in enumerate(pows):
        for j in range(len(freqs)):

            if i == 0:
                new_avgs[j] = 1

            avg_list[j].append(new_avgs[j])
            curr_avg = np.max([new_avgs[j], min_avg])

            pow_name = str(power)
            fname = f"{freqs0[j]*10:1.0f}"
            pow_list.append(power)
            if i == 0:

                new_conf = {
                    "npoints": 500,
                    "center_freq": float(freqs[j]),
                    "span": float(spans[j]) * 1.3,
                    "reps": int(curr_avg),
                }
                qick_config.update(new_conf)
                file_name = "res_" + fname + "_" + "_single"

                min_freq, data = run_freq_loop(
                    config["soc"],
                    config["soccfg"],
                    qick_config,
                    expt_path,
                    file_name,
                    plot=False,
                    progress=False,
                )
                fitparams = [min_freq, 10, 10, 0, np.max(data["amps"]), 0]
                freqs[j], q, kappa, pars = fit_resonator(
                    data, fitparams, config["slope"][j], pows[i]
                )
                spans[j] = kappa * config["span_inc"]

            new_conf = {
                "npoints": config["npoints"],
                "center_freq": float(freqs[j]),
                "pow": pows[i],
                "span": float(spans[j]),
                "reps": int(curr_avg),
            }
            qick_config.update(new_conf)

            file_name = "res_" + fname + "_" + pow_name
            min_freq, datac = fix_reps(
                config["soc"],
                config["soccfg"],
                qick_config,
                expt_path,
                file_name,
                plot=False,
                progress=False,
            )
            freqs[j] = min_freq

            amps = datac["amps"]

            amps = datac["amps"]

            # Fit data
            # f0, Qi, Qe, phi, scale, a0, slope
            fitparams = [min_freq, pars[1], pars[2], pars[3], np.max(amps), 0]
            freqs[j], q, kappa, pars = fit_resonator(
                datac, fitparams, config["slope"][j], power
            )
            pars_list[-1].append(pars)

            pin = config["power_start"] - (pars[0] - 6000) / 1000 * 5
            nph = n(pin - power * config["exp_val"], pars[0] * 1e6, q, pars[2])
            plt.figure(figsize=(4, 4))
            qi[j, i] = pars[1] * 1e4
            plt.plot(-pows, qi[j, :], "o-")
            plt.show()
            new_avgs[j] = np.round(config["avg_corr"] / nph / config["soft_avgs"])
            # Choose new span and averaging number
            print(f"Pin {pin-power:.3f}, N photons: {nph:.3g}, navg: {new_avgs[j]:1f}")

            # print(new_avg)
            spans[j] = kappa * config["span_inc"]

            plt.show()
    return pars_list


def fit_resonator(data, fitparams, slope, power):

    pars, err, pinit = fitter.fithanger(data["xpts"], data["amps"], fitparams=fitparams)
    pars, err, pinit = fitter.fithanger(data["xpts"], data["amps"], fitparams=pars)
    plot_scan(
        data["xpts"],
        data["amps"],
        data["phases"],
        pars,
        pinit,
        power,
        slope=slope,
    )
    freq_center = pars[0]
    q = 1 / (1 / pars[1] + 1 / pars[2]) * 1e4
    kappa = freq_center / q
    return freq_center, q, kappa, pars


def n(p, f, q, qc):
    return pow_res(p) * q**2 / qc / (cs.h * f**2 * np.pi)


def pow_res(p):
    return 10 ** (p / 10) * 1e-3


def do_vna_scan_segments(VNA, file_name, expt_path, cfg):

    freq_center = cfg["freq_center"]

    bandwidth = cfg["bandwidth"]
    power = cfg["power"]  # -50 #dBm
    scattering_parameter = ("S21",)  # needs to be tuple
    trace_name = ("trace1",)  # needs to be tuple

    # VNA setup
    wid = 0.35
    pt_spc = cfg["span"] * (1 - wid) / 2 / cfg["npoints1"]
    freq_sweep0 = np.linspace(
        freq_center - cfg["span"] / 2,
        freq_center - wid / 2 * cfg["span"] - pt_spc,
        cfg["npoints1"],
    )
    freq_sweep1 = np.linspace(
        freq_center - wid * cfg["span"] / 2,
        freq_center + wid * cfg["span"] / 2,
        cfg["npoints2"],
    )
    freq_sweep2 = np.linspace(
        freq_center + wid / 2 * cfg["span"] + pt_spc,
        freq_center + cfg["span"] / 2,
        cfg["npoints1"],
    )
    freq_sweep = np.concatenate((freq_sweep0, freq_sweep1, freq_sweep2))
    VNA.initialize_one_tone_spectroscopy(trace_name, scattering_parameter)
    swp_typ = "sweeptime"
    VNA.define_segment(
        0,
        freq_center - cfg["span"] / 2,
        freq_center - wid / 2 * cfg["span"] - pt_spc,
        cfg["npoints1"],
        cfg["power"],
        1 / cfg["bandwidth"],
        cfg["bandwidth"],
        set_time=swp_typ,
    )
    VNA.define_segment(
        1,
        freq_center - wid * cfg["span"] / 2,
        freq_center + wid * cfg["span"] / 2,
        cfg["npoints2"],
        cfg["power"],
        1 / cfg["bandwidth"],
        cfg["bandwidth"],
        set_time=swp_typ,
    )
    VNA.define_segment(
        2,
        freq_center + wid / 2 * cfg["span"] + pt_spc,
        freq_center + cfg["span"] / 2,
        cfg["npoints1"],
        cfg["power"],
        1 / cfg["bandwidth"],
        cfg["bandwidth"],
        set_time=swp_typ,
    )
    VNA.set_averages(cfg["averages"])
    VNA.set_averagestatus(status="on")
    VNA.set_sweeps(1)

    ## Sanity check power setting
    power_at_device = cfg["power"] - warm_att - cold_att

    VNA.measure()
    [amps, phases] = VNA.get_traces(("trace1",))[0]
    data = {
        "series": "",  # datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
        "amps": amps,
        "phases": phases,
        "freqs": freq_sweep,
        "vna_power": power,
        "power_at_device": power - warm_att - cold_att,
        "bandwidth": bandwidth,
        "averages": cfg["averages"],
        "nb_points": cfg["nb_points"],
    }
    """Saving Data"""

    with SlabFile(expt_path + "//" + file_name, "w") as f:
        # f.append_pt((vary_param + '_pts'), pt) # Flux
        f.append_line("fpts", freq_sweep)
        f.append_line("mags", amps)
        f.append_line("phases", phases)
        f.append_pt("vna_power", power)
        f.append_pt("averages", cfg["averages"])
        f.append_pt("bandwidth", bandwidth)
        f.append_pt("nb_points", cfg["nb_points"])

    #         fp1, fn1 = write_file(data, expt_path, filename=file_name)
    plot_amp(data, filepath=expt_path)
    plot_unwrapped_phase(data, filepath=expt_path)

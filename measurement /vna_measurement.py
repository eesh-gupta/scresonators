import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from ZNB import ZNB20
from VNA_funcs import *
from datamanagement import SlabFile
import fitting as fitter
import copy
from scipy.optimize import curve_fit
import scipy.constants as cs
from scresonators.fit_resonator import ana_tls
from scresonators.fit_resonator.ana_resonator import ResonatorFitter

def do_vna_scan(VNA, file_name, expt_path, cfg, spar="s21", att=0, plot=True):
    """
    Perform a VNA scan and save the data to a file.

    Parameters:
    -----------
    VNA : ZNB object
        The VNA instrument object
    file_name : str
        Name for saving the data
    expt_path : str
        Path for saving the data
    cfg : dict
        Configuration dictionary with measurement parameters:
        - freq_center: Center frequency in MHz
        - span: Frequency span in MHz
        - npoints: Number of frequency points
        - power: VNA output power in dBm
        - bandwidth: Measurement bandwidth in Hz
        - averages: Number of averages
    spar : str
        Scattering parameter (e.g., 'S21')
    att : float, optional
        Attenuation value in dB, default is 0
    plot : bool, optional
        Whether to plot the data, default is True

    Returns:
    --------
    dict
        Dictionary containing the measurement data
    """
    try:
        # Calculate frequency range
        freq_center = cfg["freq_center"]
        freq_span = cfg["span"]
        freq_start = freq_center - 0.5 * freq_span
        freq_stop = freq_center + 0.5 * freq_span

        # Generate frequency sweep points
        freq_sweep = np.linspace(freq_start, freq_stop, cfg["npoints"])

        # Get power setting
        power = cfg["power"]

        # Prepare trace and scattering parameter
        trace_name = ("trace1",)
        scattering_parameter = (spar,)

        # Configure VNA
        VNA.initialize_one_tone_spectroscopy(trace_name, scattering_parameter)
        VNA.set_startfrequency(freq_start)
        VNA.set_stopfrequency(freq_stop)
        VNA.set_points(cfg["npoints"])
        VNA.set_power(power)
        VNA.set_measBW(cfg["bandwidth"])
        VNA.set_sweeps(cfg["averages"])
        VNA.set_averages(cfg["averages"])
        VNA.set_averagestatus(status="on")
        time_expected = 1 / cfg["bandwidth"] * cfg["npoints"]*cfg["averages"]
        tstart = datetime.datetime.now()    
        # Calculate actual power at device
        power_at_device = power - att

        # Perform measurement
        VNA.measure()

        # Get measurement data
        [amps, phases] = VNA.get_traces(trace_name)[0]

        # Create data dictionary with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "series": timestamp,
            "amps": amps,
            "phases": phases,
            "freqs": freq_sweep,
            "vna_power": power,
            "power_at_device": power_at_device,
            "bandwidth": cfg["bandwidth"],
            "averages": cfg["averages"],
            "npoints": cfg["npoints"],
        }
        tfinish = datetime.datetime.now()
        print(f"Time elapsed: {(tfinish-tstart)/60} min, expected time: {time_expected/60} min")
        # Save data to file
        file_path = os.path.join(expt_path, file_name)
        with SlabFile(file_path, "w") as f:
            f.append_line("fpts", freq_sweep)
            f.append_line("mags", amps)
            f.append_line("phases", phases)
            f.append_pt("vna_power", power)
            f.append_pt("power_at_device", power_at_device)
            f.append_pt("averages", cfg["averages"])
            f.append_pt("bandwidth", cfg["bandwidth"])
            f.append_pt("npoints", cfg["npoints"])
            f.append_pt("timestamp", timestamp)

        # Plot data if requested
        if plot:
            plot_all(data, filepath=expt_path)

        return data

    except Exception as e:
        print(f"Error in do_vna_scan: {str(e)}")
        raise


def plot_scan(freq, amps, phase, pars=None, pinit=None, power=None, slope=None):

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(freq / 1e6, amps, "k.", markersize=3)
    ax[0].set_xlabel("Frequency (MHz)")
    ax[0].set_ylabel("Amplitude")
    if pars is not None:
        q = 1 / (1 / pars[1] + 1 / pars[2]) * 1e4
        qi = pars[1] * 1e4
        qc = pars[2] * 1e4
        lab = f"$Q$={q:.3g}\n $Q_i$={qi:.3g} \n $Q_c$={qc:.3g}"
        ax[0].plot(freq / 1e6, fitter.hangerS21func(freq, *pars))
        ax[0].plot(freq / 1e6, fitter.hangerS21func(freq, *pinit))
    ax[0].set_title(f"Power: {power:.1f} dB")
    if slope is None:
        phase = np.unwrap(phase)
        slope, of = np.polyfit(freq, phase, 1)
        phase_sub = phase - slope * freq - of
    else:
        phase_sub = phase - slope * freq
    ax[1].plot(
        freq / 1e6,
        np.unwrap(phase_sub) - np.mean(np.unwrap(phase_sub)),
        "k.-",
        markersize=3,
    )
    ax[1].set_xlabel("Frequency (MHz)")
    ax[1].set_ylabel("Phase")
    #ax[0].legend(loc="lower right")
    ax[0].text(0.95, 0.05, lab, transform=ax[0].transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
    ax2 = ax[0].twinx()
    if pars is not None:
        ax2.set_ylim(ax[0].get_ylim()[0] / pars[4], ax[0].get_ylim()[1] / pars[4])
    ax2.set_ylabel("Normalized Amplitude")

    ax[2].plot(amps * np.cos(phase_sub), amps * np.sin(phase_sub), "k.-")
    fig.tight_layout()
    plt.show()


def power_sweep(config, VNA):
    """
    Perform a power sweep scan using the do_vna_scan function.

    Parameters:
    -----------
    config : dict
        Configuration dictionary with measurement parameters:
        - base_path: Base directory path
        - folder: Folder name for saving data
        - freqs: List of center frequencies to scan
        - nvals: Number of power values to sweep
        - pow_inc: Power increment
        - pow_start: Starting power
        - span_inc: Span increment factor [number of linewidths for scan]
        - kappa_start: Initial kappa value
        - npoints: Number of frequency points
        - bandwidth: Measurement bandwidth in Hz
        - averages: Number of averages
        - att: Attenuation value (optional, default is 0)
    VNA : ZNB object
        The VNA instrument object

    Returns:
    --------
    list
        List of parameter lists for each frequency and power
    """

    # Create experiment path
    expt_path = os.path.join(config["base_path"], config["folder"])
    try:
        os.makedirs(expt_path)
    except:
        pass

    # Copy frequency lists to avoid modifying the original
    freqs0 = copy.deepcopy(config["freqs"])
    freqs = copy.deepcopy(config["freqs"])

    # Generate power values to sweep
    pows = np.arange(0, config["nvals"]) * config["pow_inc"] + config["pow_start"]

    # Initialize arrays for spans and averaging
    spans = config["span_inc"] * config["kappa_start"] * np.ones(len(freqs0))
    new_avgs = np.ones(len(freqs0))
    min_avg = 12

    # Initialize lists for results
    pars_list = [[] for i in range(len(freqs0))]
    avg_list = [[] for i in range(len(freqs0))]
    qi_list = [[] for i in range(len(freqs0))]
    qc_list = [[] for i in range(len(freqs0))]
    nph_list = [[] for i in range(len(freqs0))]
    q2_list = [[] for i in range(len(freqs0))]
    qc2_list = [[] for i in range(len(freqs0))]
    freq2_list = [[] for i in range(len(freqs0))]
    qi2_list = [[] for i in range(len(freqs0))]
    pars = [[] for i in range(len(freqs0))]
    pow_list = []

    # Get attenuation value (if provided)
    att = config["att"]

    # Perform power sweep
    
    for j in range(len(freqs)):
        for i, power in enumerate(pows):
            # Set initial averaging for first power point
            if i == 0:
                new_avgs[j] = 1

            # Track averaging
            avg_list[j].append(new_avgs[j])
            curr_avg = max(new_avgs[j], min_avg)

            # Create filenames
            pow_name = f"{power:.0f}"
            fname = f"{freqs0[j]*9:1.0f}"
            pow_list.append(power)

            # For first power point, do an initial scan with wider span
            if i == 0:
                # Configure VNA scan
                vna_config = {
                    "freq_center": float(freqs[j]),
                    "span": float(spans[j]) * 1.3,
                    "npoints": 800,
                    "power": power,
                    "bandwidth": config["bandwidth"],
                    "averages": 1,
                }

                # Perform VNA scan
                file_name = f"res_{fname}_single.h5"
                data = do_vna_scan(
                    VNA, file_name, expt_path, vna_config, "S21", att=att, plot=False
                )

                # Fit resonator to find center frequency and kappa
                min_freq = freqs[j]  # Initial guess
                # f0, Qi, Qe, phi, scale, a0, slope

                freqs[j], q, kappa, pars[j] = fit_resonator(data, power)
                output_path = os.path.join(expt_path)
               
                spans[j] = kappa * config["span_inc"]

            
            if i <6:
                #npoints = 2*config["npoints"]
                npoints = config["npoints"]
                span = spans[j]*1.25
            else:
                npoints = config["npoints"]
            # Configure VNA scan for this power point
            vna_config = {
                "freq_center": float(freqs[j]),
                "span": span,
                "kappa": spans[j]/config["span_inc"],
                "npoints": npoints,
                "npoints1": config["npoints1"],
                "npoints2": config["npoints2"],
                "power": power,
                "bandwidth": config["bandwidth"],
                "averages": int(curr_avg),
            }

            # Perform VNA scan
            file_name = f"res_{fname}_{pow_name}dbm.h5"
            if config["type"] == "lin":
                data = do_vna_scan(
                    VNA, file_name, expt_path, vna_config, "S21", att=att, plot=False
                )
            elif config["type"] == "single":
                data = do_vna_scan_single_point(
                    VNA, file_name, expt_path, vna_config, "S21", plot=False
                )
            else:
                data = do_vna_scan_segments(
                    VNA, file_name, expt_path, vna_config, spar="s21", plot=False
                )

            # Fit data to find resonator parameters
            min_freq = freqs[j]  # Use previous frequency as initial guess
            if i>0:
                q_old = q

            if i<6:
                fitparams = [
                min_freq,
                pars[j][1],
                pars[j][2],
                pars[j][3],
                np.max(10 ** (data["amps"] / 20))
            ]
                freqs[j], q, kappa, pars[j] = fit_resonator(data, power, fitparams)
            else:
                fitparams = [
                min_freq,
                pars[j][1],
                pars[j][3],
                np.max(10 ** (data["amps"] / 20))
            ]
                qc_best = np.mean(qc_list[j][1:5])
                freqs[j], q, kappa, pars[j] = fit_resonator(data, power, fitparams, qc_best)
            try:
                mmm=1
                # output = ResonatorFitter.fit_resonator(
                #                         data, fname, output_path, plot=True, fix_freq=False
                #                     )
                # q2 = output[0][0]
                # qc2 = output[0][1]
                # freq2 = output[0][2]
                qi2 = 1/(1/q2-1/qc2)
                q2_list[j].append(q2)
                qc2_list[j].append(qc2)
                freq2_list[j].append(freq2)
                qi2_list[j].append(qi2)
            except:
                q2=np.nan
                qc2=np.nan
                freq2=np.nan
            
            
            pars_list[j].append(pars)
            if i<6:
                qc = pars[j][2] * 1e4
            else:
                qc = qc_best
            qc_list[j].append(qc)


            # Calculate photon number (if needed)
            pin = power - config["att"]
            nph = n(pin, pars[j][0], q, qc)

            # power in, frequency in Hz, quality factor, external quality factor

            # Plot Qi vs power
            fig, ax = plt.subplots(1,2, figsize=(8,3))
            qi = pars[j][1] * 1e4
            qi_list[j].append(qi)
            nph_list[j].append(nph)
            ax[0].semilogx(nph_list[j], qi_list[j], "o-")
            #ax[0].semilogx(nph_list[j], qi2_list[j], "o-")
            # Fit Qi vs power to an exponential
            

            if len(qi_list[j]) > 5:
                try:
                    q_fitn = lambda n, Qtls0, Qoth, nc, beta: ana_tls.Qtotn(n,0.04, min_freq, Qtls0, Qoth, nc, beta)
                    p = [np.min(qi_list[j]), np.max(qi_list[j]), 3, 0.4]
                    p, err = curve_fit(q_fitn, nph_list[j], qi_list[j], p0=p)
                    ax[0].plot(nph_list[j], q_fitn(nph_list[j], *p), 'r--', label='Exp fit')
                    print(p)
                except Exception as e:
                    print(f"Fit failed: {str(e)}")
            
            
            ax[0].set_xlabel("Number of Photons")
            ax[0].set_ylabel("Internal Quality Factor ($Q_i$)")
            ax[0].set_title(f"Frequency: {freqs[j]/1e9:.5f} GHz")
            ax[1].semilogx(nph_list[j], qc_list[j], "o-")
            #ax[1].semilogx(nph_list[j], qc2_list[j], "o-")
            plt.show()

            # Calculate new averaging based on photon number
            # avg_corr is a fudge factor
            if i>0: 
                q_adg = q/q_old
            else:
                q_adg = 0.9
            
            print(f"Q_adg: {q_adg:.3f}")
            
            if "avg_corr" in config:
                new_avgs[j] = np.round(config["avg_corr"] / nph/q_adg**2)
                print(
                    f"Pin {power-config['att']:.1f}, N photons: {nph:.3g}, navg: {int(new_avgs[j])}"
                )

            # Update span for next measurement
            spans[j] = kappa * config["span_inc"]

    return pars_list


def fit_resonator(data, power, fitparams=None, qc=None):

    # Convert amplitude from dB to linear scale
    amps_linear = 10 ** (data["amps"] / 20)
    # f0, Qi, Qe, phi, scale, a0, slope # Qi/Qe in units of 10k 
    if qc is not None:
        hangerfit = lambda f, f0, qi, phi, scale: fitter.hangerS21func(f, f0, qi, qc/1e4, phi, scale)
        
        pars, err = curve_fit(hangerfit, data["freqs"], amps_linear, p0=fitparams)
        freq_center = pars[0]
        q = 1 / (1 / pars[1] / 1e4 + 1 / qc)
        kappa = freq_center / q
        r2=fitter.get_r2(data["freqs"], amps_linear, hangerfit, pars)
        err = np.sqrt(np.diag(err))
        print('f error: ', err[0], 'Qi error: ', err[1], 'phi error: ', err[3], 'scale error: ', err[4])
        pars = [pars[0], pars[1], qc/1e4, pars[2], pars[3]]
        fitparams = [pars[0], pars[1], qc/1e4, pars[3], pars[4]]
        
        print(f"R2: {r2}")

    else:        
        if fitparams is None:
            min_freq = data["freqs"][np.argmin(amps_linear)]
            fitparams = [min_freq, 100, 100, 0, np.max(amps_linear)]

        pars, err, pinit = fitter.fithanger(data["freqs"], amps_linear, fitparams=fitparams)

        pars, err, pinit = fitter.fithanger(data["freqs"], amps_linear, fitparams=pars)
        
        freq_center = pars[0]
        q = 1 / (1 / pars[1] + 1 / pars[2]) * 1e4
        kappa = freq_center / q
    plot_scan(
            data["freqs"],
            amps_linear,
            data["phases"],
            pars,
            fitparams,
            power,
        )
    
    return freq_center, q, kappa, pars


def n(p, f, q, qc):
    return pow_res(p) * q**2 / qc / (cs.h * f**2 * np.pi)


def pow_res(p):
    return 10 ** (p / 10) * 1e-3




def get_default_power_sweep_config(custom_config=None):
    """
    Get default configuration for power_sweep function.

    Parameters:
    -----------
    custom_config : dict, optional
        Dictionary with custom configuration values to override defaults

    Returns:
    --------
    dict
        Configuration dictionary with default values for power_sweep
    """
    # Define default configuration
    default_config = {
        # File paths
        "base_path": "./data",
        "folder": f"power_sweep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        # Frequency settings
        "freqs": np.array([6]) * 1e9,  # Default center frequency in Hz
        "span_inc": 8,  # Span as multiple of linewidth
        "kappa_start": 30000,  # Initial linewidth estimate in Hz
        # Power sweep settings
        "nvals": 18,  # Number of power points
        "pow_start": -5,  # Starting power in dBm
        "pow_inc": -5,  # Power increment in dB
        # Measurement settings
        "npoints": 201,  # Number of frequency points
        "npoints1": 10,
        "npoints2": 27,
        "bandwidth": 100,  # Measurement bandwidth in Hz
        "averages": 1,  # Number of averages
        "att": 60,  # Attenuation in dB
        "type": "lin",
        "freq_0": 6e9,
        "db_slope":2.2,
        # Analysis settings
        "avg_corr": 2e5,  # Correction factor for averaging
    }
    if custom_config is not None and "type" in custom_config and custom_config["type"] == "single":
        default_config['npoints'] = 31

    # Override defaults with custom values if provided
    if custom_config is not None:
        for key, value in custom_config.items():
            default_config[key] = value

    return default_config


def do_vna_scan_segments(
    VNA, file_name, expt_path, cfg, spar="s21", warm_att=0, cold_att=0, plot=True
):
    """
    Perform a VNA scan with segmented frequency ranges and save the data to a file.

    This function divides the frequency span into three segments:
    1. Lower frequency range with fewer points
    2. Center frequency range with more points (higher resolution)
    3. Upper frequency range with fewer points

    Parameters:
    -----------
    VNA : ZNB object
        The VNA instrument object
    file_name : str
        Name for saving the data
    expt_path : str
        Path for saving the data
    cfg : dict
        Configuration dictionary with measurement parameters:
        - freq_center: Center frequency in MHz
        - span: Frequency span in MHz
        - npoints1: Number of frequency points for outer segments
        - npoints2: Number of frequency points for center segment
        - power: VNA output power in dBm
        - bandwidth: Measurement bandwidth in Hz
        - averages: Number of averages
        - npoints: Total number of points (for metadata)
    warm_att : float, optional
        Warm attenuation value in dB, default is 0
    cold_att : float, optional
        Cold attenuation value in dB, default is 0
    plot : bool, optional
        Whether to plot the data, default is True

    Returns:
    --------
    dict
        Dictionary containing the measurement data
    """
    try:
        # Get frequency center and bandwidth
        freq_center = cfg["freq_center"]
        bandwidth = cfg["bandwidth"]
        power = cfg["power"]


        # Prepare trace and scattering parameter
        scattering_parameter = (spar,)
        trace_name = ("trace1",)

        # Define segment width ratio (center segment width / total span)
        wid = 0.2

        # Calculate point spacing for smooth transitions between segments
        pt_spc = cfg["span"] * (1 - wid) / 2 / cfg["npoints1"]

        # Generate frequency points for each segment
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

        # Combine all frequency points
        freq_sweep = np.concatenate((freq_sweep0, freq_sweep1, freq_sweep2))

        # Initialize VNA
        VNA.initialize_one_tone_spectroscopy_seg(trace_name, scattering_parameter)

        # Define sweep type
        swp_typ = "dwell"

        # Configure VNA segments
        # Segment 0: Lower frequency range
        npoints = cfg["npoints1"]+cfg["npoints2"]+cfg["npoints1"]
        time_expected = 1 / cfg["bandwidth"] * npoints*cfg["averages"]
        print(f"expected time: {time_expected / 60} min")
        t = 1 / cfg["bandwidth"]/6
        tstart = datetime.datetime.now()    
        VNA.define_segment(
            1,
            freq_center - cfg["span"] / 2,
            freq_center - wid / 2 * cfg["span"] - pt_spc,
            cfg["npoints1"],
            cfg["power"],
            t,
            cfg["bandwidth"],
            set_time=swp_typ,
        )

        # Segment 1: Center frequency range (higher resolution)
        VNA.define_segment(
            2,
            freq_center - wid * cfg["span"] / 2,
            freq_center + wid * cfg["span"] / 2,
            cfg["npoints2"],
            cfg["power"],
            t,
            cfg["bandwidth"],
            set_time=swp_typ,
        )

        # Segment 2: Upper frequency range
        VNA.define_segment(
            3,
            freq_center + wid / 2 * cfg["span"] + pt_spc,
            freq_center + cfg["span"] / 2,
            cfg["npoints1"],
            cfg["power"],
            t,
            cfg["bandwidth"],
            set_time=swp_typ,
        )

        # Set averaging parameters
        VNA.set_averages(cfg["averages"])
        VNA.set_averagestatus(status="on")
        VNA.set_sweeps(cfg["averages"])

        # Calculate actual power at device
        power_at_device = cfg["power"] - warm_att - cold_att

        # Perform measurement
        VNA.measure()

        # Get measurement data
        [amps, phases] = VNA.get_traces(trace_name)[0]
        tfinish = datetime.datetime.now()
        elapsed_time = (tfinish - tstart).total_seconds()
        print(f"Time elapsed: {elapsed_time / 60} min")
        
        # Create data dictionary with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "series": timestamp,
            "amps": amps,
            "phases": phases,
            "freqs": freq_sweep,
            "vna_power": power,
            "power_at_device": power_at_device,
            "bandwidth": bandwidth,
            "averages": cfg["averages"],
            "npoints": cfg["npoints"],
        }

        # Save data to file
        file_path = os.path.join(expt_path, file_name)
        with SlabFile(file_path, "w") as f:
            f.append_line("freqs", freq_sweep)
            f.append_line("amps", amps)
            f.append_line("phases", phases)
            f.append_pt("vna_power", power)
            f.append_pt("power_at_device", power_at_device)
            f.append_pt("averages", cfg["averages"])
            f.append_pt("bandwidth", bandwidth)
            f.append_pt("npoints", cfg["npoints"])
            f.append_pt("timestamp", timestamp)

        # Plot data if requested
        if plot:
            plot_all(data, filepath=expt_path)

        return data

    except Exception as e:
        print(f"Error in do_vna_scan_segments: {str(e)}")
        raise

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

    N = config["npoints"]
    df = config["span"]
    w = df / config["kappa"]*config['kappa_inc']
    at = np.arctan(2 * w / (1 - w**2)) + np.pi
    R = w / np.tan(at / 2)
    fr = config["freq_center"]
    n = np.arange(N) - N / 2 + 1 / 2
    flist=fr + R * df / (2 * w) * np.tan(n / (N - 1) * at)
    flist_lin = -np.arange(3,1,-1)*df/N*2+config["freq_center"]-config["span"]/2
    flist_linp = np.arange(1,3)*df/N*2+config["freq_center"]+config["span"]/2
    flist = np.concatenate([flist_lin, flist, flist_linp])
    return flist


def do_vna_scan_single_point(
    VNA, file_name, expt_path, cfg, spar="s21", warm_att=0, cold_att=0, plot=True
):
    """
    Perform a VNA scan with segmented frequency ranges and save the data to a file.

    This function divides the frequency span into three segments:
    1. Lower frequency range with fewer points
    2. Center frequency range with more points (higher resolution)
    3. Upper frequency range with fewer points

    Parameters:
    -----------
    VNA : ZNB object
        The VNA instrument object
    file_name : str
        Name for saving the data
    expt_path : str
        Path for saving the data
    cfg : dict
        Configuration dictionary with measurement parameters:
        - freq_center: Center frequency in MHz
        - span: Frequency span in MHz
        - npoints: Number of frequency points for outer segments
        - power: VNA output power in dBm
        - bandwidth: Measurement bandwidth in Hz
        - averages: Number of averages
        - npoints: Total number of points (for metadata)
    warm_att : float, optional
        Warm attenuation value in dB, default is 0
    cold_att : float, optional
        Cold attenuation value in dB, default is 0
    plot : bool, optional
        Whether to plot the data, default is True

    Returns:
    --------
    dict
        Dictionary containing the measurement data
    """
    try:
        # Get frequency center and bandwidth
        
        bandwidth = cfg["bandwidth"]
        power = cfg["power"]
        cfg["kappa_inc"] = 1.1

        freq_list = get_homophase(cfg)

        # Prepare trace and scattering parameter
        scattering_parameter = (spar,)
        trace_name = ("trace1",)
        
        # Initialize VNA
        # start by deleting all previous segments. 
        #VNA.delete_segments()
        #VNA.reset()
        VNA.initialize_one_tone_spectroscopy_seg(trace_name, scattering_parameter)

        # Define sweep type
        swp_typ = "dwell"

        # Configure VNA segments
        # Segment 0: Lower frequency range
        time_expected = 1 / cfg["bandwidth"] * cfg['npoints']*cfg["averages"]
        t = 1 / cfg["bandwidth"]/6
        tstart = datetime.datetime.now()    
        
        for i, freq in enumerate(freq_list):
            VNA.define_segment(
                i+1,
                freq,
                freq,
                1,
                cfg["power"],
                t,
                cfg["bandwidth"],
                set_time=swp_typ,
            )

        # Set averaging parameters
        VNA.set_averages(cfg["averages"])
        VNA.set_averagestatus(status="on")
        VNA.set_sweeps(cfg["averages"])

        # Calculate actual power at device
        power_at_device = cfg["power"] - warm_att - cold_att

        # Perform measurement
        VNA.measure()

        # Get measurement data
        [amps, phases] = VNA.get_traces(trace_name)[0]
        tfinish = datetime.datetime.now()
        elapsed_time = (tfinish - tstart).total_seconds()
        print(f"Time elapsed: {elapsed_time / 60:.2} min, expected time: {time_expected / 60:.2} min")
        
        # Create data dictionary with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "series": timestamp,
            "amps": amps,
            "phases": phases,
            "freqs": freq_list,
            "vna_power": power,
            "power_at_device": power_at_device,
            "bandwidth": bandwidth,
            "averages": cfg["averages"],
            "npoints": cfg["npoints"],
        }

        # Save data to file
        file_path = os.path.join(expt_path, file_name)
        
        with SlabFile(file_path, "w") as f:
            f.append_line("fpts", freq_list)
            f.append_line("mags", amps)
            f.append_line("phases", phases)
            f.append_pt("vna_power", power)
            f.append_pt("averages", cfg["averages"])
            f.append_pt("bandwidth", bandwidth)
            f.append_pt("npoints", cfg["npoints"])
            f.append_pt("timestamp", timestamp)
#           f.append_pt["attenuation"] = {"warm": warm_att, "cold": cold_att}

        # Plot data if requested
        if plot:
            plot_all(data, filepath=expt_path)

        return data

    except Exception as e:
        print(f"Error in do_vna_scan_segments: {str(e)}")
        raise


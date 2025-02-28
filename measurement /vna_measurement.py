import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from ZNB import ZNB20
from VNA_funcs import *
from datamanagement import SlabFile
import fitting as fitter
import copy
import scipy.constants as cs


def do_vna_scan(VNA, file_name, expt_path, cfg, spar, att=0, plot=True):
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
        - nb_points: Number of frequency points
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
        freq_sweep = np.linspace(freq_start, freq_stop, cfg["nb_points"])

        # Get power setting
        power = cfg["power"]

        # Prepare trace and scattering parameter
        trace_name = ("trace1",)
        scattering_parameter = (spar,)

        # Configure VNA
        VNA.initialize_one_tone_spectroscopy(trace_name, scattering_parameter)
        VNA.set_startfrequency(freq_start)
        VNA.set_stopfrequency(freq_stop)
        VNA.set_points(cfg["nb_points"])
        VNA.set_power(power)
        VNA.set_measBW(cfg["bandwidth"])
        VNA.set_sweeps(cfg["averages"])
        VNA.set_averages(cfg["averages"])
        VNA.set_averagestatus(status="on")

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
            "nb_points": cfg["nb_points"],
        }

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
            f.append_pt("nb_points", cfg["nb_points"])
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
        lab = f"$Q$={q:.3g},\n $Q_i$={qi:.3g}"
        ax[0].plot(freq / 1e6, fitter.hangerS21func_sloped(freq, *pars), label=lab)
        ax[0].plot(freq / 1e6, fitter.hangerS21func_sloped(freq, *pinit))
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
    ax[0].legend(loc="lower right")
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
    min_avg = 1

    # Initialize lists for results
    pars_list = [[] for i in range(len(freqs0))]
    avg_list = [[] for i in range(len(freqs0))]
    qi_list = [[] for i in range(len(freqs0))]
    pars = [[] for i in range(len(freqs0))]
    pow_list = []

    # Get attenuation value (if provided)
    att = config["att"]

    # Perform power sweep
    for i, power in enumerate(pows):
        for j in range(len(freqs)):
            # Set initial averaging for first power point
            if i == 0:
                new_avgs[j] = 1

            # Track averaging
            avg_list[j].append(new_avgs[j])
            curr_avg = max(new_avgs[j], min_avg)

            # Create filenames
            pow_name = f"{power:.1f}"
            fname = f"{freqs0[j]*10:1.0f}"
            pow_list.append(power)

            # For first power point, do an initial scan with wider span
            if i == 0:
                # Configure VNA scan
                vna_config = {
                    "freq_center": float(freqs[j]),
                    "span": float(spans[j]) * 1.3,
                    "nb_points": 800,
                    "power": power,
                    "bandwidth": config["bandwidth"],
                    "averages": int(curr_avg),
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
                spans[j] = kappa * config["span_inc"]

            # Configure VNA scan for this power point
            vna_config = {
                "freq_center": float(freqs[j]),
                "span": float(spans[j]),
                "nb_points": config["npoints"],
                "power": power,
                "bandwidth": config["bandwidth"],
                "averages": int(curr_avg),
            }

            # Perform VNA scan
            file_name = f"res_{fname}_{pow_name}dbm.h5"
            data = do_vna_scan(
                VNA, file_name, expt_path, vna_config, "S21", att=att, plot=False
            )

            # Fit data to find resonator parameters
            min_freq = freqs[j]  # Use previous frequency as initial guess
            fitparams = [
                min_freq,
                pars[j][1],
                pars[j][2],
                pars[j][3],
                np.max(10 ** (data["amps"] / 20)),
                0,
            ]
            freqs[j], q, kappa, pars[j] = fit_resonator(data, power, fitparams)
            pars_list[j].append(pars)

            # Calculate photon number (if needed)
            pin = power - config["att"]
            nph = n(pin, pars[j][0], q, pars[j][2] * 1e4)

            # power in, frequency in Hz, quality factor, external quality factor

            # Plot Qi vs power
            plt.figure(figsize=(4, 3))
            qi = pars[j][1] * 1e4
            qi_list[j].append(qi)
            plt.plot(pows[: i + 1], qi_list[j], "o-")
            plt.xlabel("Power (dBm)")
            plt.ylabel("Internal Quality Factor (Qi)")
            plt.title(f"Frequency: {freqs[j]/1e9:.5f} GHz")
            plt.show()

            # Calculate new averaging based on photon number
            # avg_corr is a fudge factor
            if "avg_corr" in config:
                new_avgs[j] = np.round(config["avg_corr"] / nph)
                print(
                    f"Pin {power-config['att']:.3f}, N photons: {nph:.3g}, navg: {new_avgs[j]:1f}"
                )

            # Update span for next measurement
            spans[j] = kappa * config["span_inc"]

    return pars_list


def fit_resonator(data, power, fitparams=None):

    # Convert amplitude from dB to linear scale
    amps_linear = 10 ** (data["amps"] / 20)
    if fitparams is None:
        min_freq = data["freqs"][np.argmin(amps_linear)]
        fitparams = [min_freq, 100, 100, 0, np.max(amps_linear), 0]
    pars, err, pinit = fitter.fithanger(data["freqs"], amps_linear, fitparams=fitparams)

    pars, err, pinit = fitter.fithanger(data["freqs"], amps_linear, fitparams=pars)
    plot_scan(
        data["freqs"],
        amps_linear,
        data["phases"],
        pars,
        fitparams,
        power,
    )
    freq_center = pars[0]
    q = 1 / (1 / pars[1] + 1 / pars[2]) * 1e4
    kappa = freq_center / q
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
        "bandwidth": 300,  # Measurement bandwidth in Hz
        "averages": 1,  # Number of averages
        "att": 60,  # Attenuation in dB
        # Analysis settings
        "avg_corr": 2e7,  # Correction factor for averaging
    }

    # Override defaults with custom values if provided
    if custom_config is not None:
        for key, value in custom_config.items():
            default_config[key] = value

    return default_config


def example_power_sweep():
    """
    Example function demonstrating how to use power_sweep with default configuration.

    This function shows how to:
    1. Get default configuration
    2. Customize specific parameters
    3. Run power sweep with a VNA

    Note: This is an example and not meant to be run directly as it requires
    a connected VNA instrument.
    """
    # Import required modules
    import os

    # Get default configuration
    config = get_default_power_sweep_config()

    # Customize configuration as needed
    custom_config = {
        "base_path": os.path.expanduser("~/data"),  # Save to user's home directory
        "freqs": [5800, 6200],  # Scan two frequencies
        "nvals": 18,  # Use 5 power points
        "pow_start": -10,  # Start at -40 dBm
        "pow_inc": 5,  # 5 dB steps
        "npoints": 401,  # More frequency points for better resolution
        "averages": 1,  # More averages for better SNR
    }

    # Update default config with custom values
    config = get_default_power_sweep_config(custom_config)

    # Connect to VNA (example only - actual connection depends on your setup)
    try:
        # This is just an example - in practice you would use your actual VNA connection
        VNA = ZNB20("TCPIP0::192.168.1.1::INSTR")  # Example IP address

        # Run power sweep
        results = power_sweep(config, VNA)

        print(f"Power sweep completed with {len(results)} frequency results")
        print(f"Each frequency has {len(results[0])} power points")

        # Process results as needed

    except Exception as e:
        print(f"Error in example_power_sweep: {str(e)}")
        print("Note: This example requires a connected VNA instrument")


def do_vna_scan_segments(
    VNA, file_name, expt_path, cfg, warm_att=0, cold_att=0, plot=True
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
        - nb_points: Total number of points (for metadata)
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
        scattering_parameter = ("S21",)
        trace_name = ("trace1",)

        # Define segment width ratio (center segment width / total span)
        wid = 0.35

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
        VNA.initialize_one_tone_spectroscopy(trace_name, scattering_parameter)

        # Define sweep type
        swp_typ = "sweeptime"

        # Configure VNA segments
        # Segment 0: Lower frequency range
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

        # Segment 1: Center frequency range (higher resolution)
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

        # Segment 2: Upper frequency range
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

        # Set averaging parameters
        VNA.set_averages(cfg["averages"])
        VNA.set_averagestatus(status="on")
        VNA.set_sweeps(1)

        # Calculate actual power at device
        power_at_device = cfg["power"] - warm_att - cold_att

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
            "bandwidth": bandwidth,
            "averages": cfg["averages"],
            "nb_points": cfg["nb_points"],
        }

        # Save data to file
        file_path = os.path.join(expt_path, file_name)
        with SlabFile(file_path, "w") as f:
            f.append_line("fpts", freq_sweep)
            f.append_line("mags", amps)
            f.append_line("phases", phases)
            f.append_pt("vna_power", power)
            f.append_pt("power_at_device", power_at_device)
            f.append_pt("averages", cfg["averages"])
            f.append_pt("bandwidth", bandwidth)
            f.append_pt("nb_points", cfg["nb_points"])
            f.append_pt("timestamp", timestamp)

        # Plot data if requested
        if plot:
            plot_amp(data, filepath=expt_path)
            plot_unwrapped_phase(data, filepath=expt_path)

        return data

    except Exception as e:
        print(f"Error in do_vna_scan_segments: {str(e)}")
        raise

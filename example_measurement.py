"""
Example script demonstrating how to use the scresonators measurement module with the new namespace structure.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import datetime

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath("."))


# Define the get_default_power_sweep_config function directly in this script
# to avoid importing modules with external dependencies
def get_default_power_sweep_config(custom_config=None):
    """
    Get default configuration for power_sweep function.
    """
    # Define default configuration
    default_config = {
        # File paths
        "base_path": "./data",
        "folder": f"power_sweep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        # Frequency settings
        "freqs": np.array([6]) * 1e9,  # Default center frequency in Hz
        "span_inc": 10,  # Span as multiple of linewidth
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
        "freq_0": 6,
        "db_slope": 4,
        # Analysis settings
        "avg_corr": 1e6,  # Correction factor for averaging
    }

    # Override defaults with custom values if provided
    if custom_config is not None:
        for key, value in custom_config.items():
            default_config[key] = value

    return default_config


# Get default configuration for power sweep
config = get_default_power_sweep_config()

# Modify some parameters for demonstration
config["freqs"] = np.array([5.05]) * 1e9  # Center frequency in Hz
config["nvals"] = 5  # Number of power points
config["pow_start"] = -10  # Starting power in dBm
config["pow_inc"] = -5  # Power increment in dB
config["folder"] = "example_measurement"  # Folder for saving data

# Print the configuration
print("Power sweep configuration:")
for key, value in config.items():
    if isinstance(value, np.ndarray):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {value}")

# In a real scenario, you would use the VNA object to perform the measurement:
# from scresonators.measurement.ZNB import ZNB20
# VNA = ZNB20("TCPIP0::192.168.1.1::INSTR")
# result = power_sweep_v2(config, VNA)

# Since we don't have a VNA connected, we'll just simulate some data
print("\nSimulating power sweep data...")
powers = np.arange(0, config["nvals"]) * config["pow_inc"] + config["pow_start"]
print(f"Power points: {powers} dBm")

# Simulate Q vs power data
q_internal = 50000 * (1 - 0.1 * np.exp(-powers / 20))
q_coupling = 20000 * np.ones_like(powers)
q_total = 1 / (1 / q_internal + 1 / q_coupling)

# Plot the simulated data
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(powers, q_internal, "o-", label="Q internal")
plt.plot(powers, q_coupling, "s-", label="Q coupling")
plt.plot(powers, q_total, "^-", label="Q total")
plt.xlabel("Power (dBm)")
plt.ylabel("Quality Factor")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
photon_numbers = 10 ** (powers / 10) * 1e-3 * q_total**2 / q_coupling
plt.semilogx(photon_numbers, q_internal, "o-")
plt.xlabel("Photon Number")
plt.ylabel("Internal Quality Factor")
plt.grid(True)

plt.tight_layout()
plt.savefig("power_sweep_example.png")
plt.close()

print("Example completed successfully!")
print("Check 'power_sweep_example.png' for the plot.")

# Scresonators Namespace Package

This repository has been restructured as a namespace package to improve organization and make imports more pythonic.

## Changes Made

1. Created a namespace package structure with `scresonators` as the top-level namespace
2. Created two subpackages: `fit_resonator` and `measurement`
3. Copied the Python files from the original directories to the new namespace package directories
4. Updated the imports in the files to use the namespace package structure
5. Updated the `setup.py` file to use the namespace package structure
6. Created example scripts that demonstrate how to use the namespace package structure

## Directory Structure

```
scresonators/
├── __init__.py
├── fit_resonator/
│   ├── __init__.py
│   ├── ana_resonator.py
│   ├── ana_tls.py
│   ├── cavity_functions.py
│   ├── check_data.py
│   ├── fit.py
│   ├── plot.py
│   ├── pyCircFit_v3.py
│   └── resonator.py
└── measurement/
    ├── __init__.py
    ├── VNA_funcs.py
    ├── ZNB.py
    ├── datamanagement.py
    ├── fitting.py
    ├── handy.py
    ├── resonator_meas.py
    └── vna_measurement.py
```

## Installation

To install the package, you can use pip:

```bash
pip install -e .
```

This will install the package in development mode, allowing you to make changes to the code and have them immediately reflected in your Python environment.

## Usage

### Fit Resonator

The `fit_resonator` subpackage provides tools for analyzing and fitting resonator data. Here's a simple example:

```python
import numpy as np
import matplotlib.pyplot as plt
from scresonators.fit_resonator.resonator import Resonator, from_columns

# Create some synthetic data
freqs = np.linspace(5e9, 5.1e9, 1000)
center_freq = 5.05e9
Q = 10000
Qc = 15000

# Calculate the response
delta = (freqs - center_freq) / center_freq
response = 1 - (Q / Qc) / (1 + 2j * Q * delta)

# Convert to amplitude and phase
amps = 20 * np.log10(np.abs(response))
phases = np.angle(response)

# Create a Resonator object
resonator = Resonator()
resonator.filename = "example_resonator"
resonator.outputpath = "./"
resonator.from_columns(freqs, amps, phases)

# Set up the fitting method
resonator.fit_method(
    method="DCM", 
    MC_iteration=3, 
    MC_rounds=100, 
    preprocess_method="circle",
    manual_init=None
)

# Perform the fit
output = resonator.fit(plot="png")

# Print the results
print(f"Q = {output[0][0]:.0f}")
print(f"Qc = {output[0][1]:.0f}")
print(f"Frequency = {output[0][2]/1e9:.6f} GHz")
print(f"Phase = {output[0][3]:.6f} rad")
```

### Measurement

The `measurement` subpackage provides tools for measurement and data acquisition. Here's a simple example:

```python
import numpy as np
from scresonators.measurement.vna_measurement import get_default_power_sweep_config, power_sweep_v2
from scresonators.measurement.ZNB import ZNB20

# Get default configuration for power sweep
config = get_default_power_sweep_config()

# Modify some parameters
config["freqs"] = np.array([5.05]) * 1e9  # Center frequency in Hz
config["nvals"] = 5  # Number of power points
config["pow_start"] = -10  # Starting power in dBm
config["pow_inc"] = -5  # Power increment in dB
config["folder"] = "example_measurement"  # Folder for saving data

# Connect to the VNA
VNA = ZNB20("TCPIP0::192.168.1.1::INSTR")

# Perform the power sweep
result = power_sweep_v2(config, VNA)
```

## Examples

For more examples, see the `example_namespace.py` and `example_measurement.py` files in the root directory.

"""
Example script demonstrating how to use the scresonators package with the new namespace structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from scresonators.fit_resonator.resonator import Resonator, from_columns

# Create some synthetic data for demonstration
freqs = np.linspace(5e9, 5.1e9, 1000)  # Frequency range from 5 to 5.1 GHz
center_freq = 5.05e9  # Resonance at 5.05 GHz
Q = 10000  # Quality factor
Qc = 15000  # Coupling quality factor

# Calculate the response using a simple model
delta = (freqs - center_freq) / center_freq
response = 1 - (Q / Qc) / (1 + 2j * Q * delta)

# Convert to amplitude and phase
amps = 20 * np.log10(np.abs(response))  # in dB
phases = np.angle(response)  # in radians

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
    manual_init=None,
)

# Perform the fit
output = resonator.fit(plot="png")

# Print the results
print("Fit results:")
print(f"Q = {output[0][0]:.0f}")
print(f"Qc = {output[0][1]:.0f}")
print(f"Frequency = {output[0][2]/1e9:.6f} GHz")
print(f"Phase = {output[0][3]:.6f} rad")

# Plot the data and fit
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(freqs / 1e9, amps, "b-", label="Data")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Amplitude (dB)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(freqs / 1e9, phases, "r-", label="Data")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Phase (rad)")
plt.legend()

plt.tight_layout()
plt.savefig("resonator_fit_example.png")
plt.close()

print("Example completed successfully!")
print("Check 'resonator_fit_example.png' for the plot.")

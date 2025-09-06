"""
Scresonators: A namespace package for measuring and fitting superconducting resonator data.

This package provides tools for:
- Fitting resonator data using various methods (DCM, INV, CPZM, etc.)
- Measuring resonator properties with VNA and other instruments
- Plotting and visualizing resonator data

Subpackages:
- fit_resonator: Resonator fitting and analysis tools
- measurement: Measurement and data acquisition tools
- plotting: Plotting and visualization utilities
"""

__version__ = "0.1.0"
__author__ = "Boulder Cryogenic Quantum Testbed"

# Import key classes and functions for convenience
try:
    from .fit_resonator.resonator import Resonator, FitMethod
    from .fit_resonator import cavity_functions

    # Import submodules as part of the namespace
    from . import fit_resonator
    from . import measurement
    from . import plotting
except ImportError:
    # Handle case where submodules might not be available
    pass

__all__ = [
    "Resonator",
    "FitMethod",
    "cavity_functions",
    "fit_resonator",
    "measurement",
    "plotting",
]

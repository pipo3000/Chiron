"""
Minimal stub for CosinorPy module.
This is a compatibility shim to allow cosinorage to load when CosinorPy is not available.
Note: CosinorPy features that depend on statsmodels may not work.
"""

# Import the cosinor1 submodule so it's available as CosinorPy.cosinor1
from . import cosinor1

__version__ = "0.0.0"


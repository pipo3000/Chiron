"""
Minimal stub for CosinorPy module.
This is a compatibility shim to allow cosinorage to load when CosinorPy is not available.
Note: CosinorPy features that depend on statsmodels may not work.
"""

import sys

# Import the cosinor1 submodule so it's available as CosinorPy.cosinor1
from . import cosinor1

# Make sure cosinor1 is accessible
cosinor1_module = cosinor1

print(f"[CosinorPy.__init__] Module loaded, cosinor1 available: {hasattr(cosinor1, 'fit_cosinor')}", file=sys.stderr)

__version__ = "0.0.0"


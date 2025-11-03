"""
Minimal stub for CosinorPy.cosinor1 module.
"""


def fit_cosinor(*args, **kwargs):
    """
    Stub function for fit_cosinor.
    The cosinorage code unpacks this as: fit, _, _, statistics = cosinor1.fit_cosinor(...)
    So we need to return a tuple of 4 elements:
    1. fit object (with .fittedvalues attribute)
    2. _ (unused, can be None)
    3. _ (unused, can be None)  
    4. statistics dict (with 'values' key containing [mesor, amplitude, acrophase])
    """
    import numpy as np
    
    # Determine data length for fitted_values
    data_length = 1440  # default
    if len(args) > 1 and hasattr(args[1], '__len__'):
        data_length = len(args[1])
    
    # Create a fit object with fittedvalues attribute
    class FitObject:
        def __init__(self, length):
            self.fittedvalues = np.zeros(length)
    
    fit = FitObject(data_length)
    
    # Statistics dict with 'values' containing [mesor, amplitude, acrophase]
    statistics = {
        'values': [0.0, 0.0, 0.0]  # [mesor, amplitude, acrophase]
    }
    
    # Return tuple of 4 elements as expected: fit, _, _, statistics
    return (fit, None, None, statistics)


class Cosinor:
    """Stub class for Cosinor."""
    def __init__(self, *args, **kwargs):
        pass
    
    def fit(self, *args, **kwargs):
        return fit_cosinor(*args, **kwargs)


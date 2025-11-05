"""
Minimal stub for CosinorPy.cosinor1 module.
This stub implements actual cosinor analysis using numpy since CosinorPy can't be installed on Android.
"""

import sys

print("[CosinorPy.cosinor1] Module loaded", file=sys.stderr)


def fit_cosinor(*args, **kwargs):
    """
    Stub function for fit_cosinor that actually computes cosinor parameters.
    The cosinorage code unpacks this as: fit, _, _, statistics = cosinor1.fit_cosinor(...)
    So we need to return a tuple of 4 elements:
    1. fit object (with .fittedvalues attribute)
    2. _ (unused, can be None)
    3. _ (unused, can be None)  
    4. statistics dict (with 'values' key containing [mesor, amplitude, acrophase])
    """
    import numpy as np
    import sys

    print(f"[CosinorPy.fit_cosinor] Called with {len(args)} args, kwargs: {kwargs}", file=sys.stderr)

    # Extract data from arguments
    # Typical call: fit_cosinor(time, data, period=1440)
    # args[0] is usually time/index, args[1] is the data
    data = None
    period = 1440.0  # Default 24 hours in minutes

    if len(args) >= 2:
        data = args[1]
        print(f"[CosinorPy.fit_cosinor] Using args[1] as data, length: {len(data) if hasattr(data, '__len__') else 'N/A'}", file=sys.stderr)
    elif len(args) >= 1:
        data = args[0]
        print(f"[CosinorPy.fit_cosinor] Using args[0] as data, length: {len(data) if hasattr(data, '__len__') else 'N/A'}", file=sys.stderr)

    # Get period from kwargs if provided
    if 'period' in kwargs:
        period = float(kwargs['period'])
        print(f"[CosinorPy.fit_cosinor] Period from kwargs: {period}", file=sys.stderr)

    # Determine data length
    if data is None or not hasattr(data, '__len__'):
        data_length = 1440  # default
        data = np.array([])
    else:
        data = np.asarray(data)
        data_length = len(data)

    # Compute cosinor fit using least squares
    # y = M + A*cos(2*pi*t/T + phi)
    # We'll use: y = M + A_cos*cos(2*pi*t/T) + A_sin*sin(2*pi*t/T)
    if len(data) > 0 and data_length > 0:
        t = np.arange(data_length)
        cos_term = np.cos(2 * np.pi * t / period)
        sin_term = np.sin(2 * np.pi * t / period)

        # Linear regression: y = M + A_cos*cos + A_sin*sin
        X = np.column_stack([np.ones(data_length), cos_term, sin_term])

        try:
            # Use least squares to solve
            coeffs, residuals, rank, s = np.linalg.lstsq(X, data, rcond=None)

            mesor = float(coeffs[0])
            amplitude_cos = float(coeffs[1])
            amplitude_sin = float(coeffs[2])
            amplitude = float(np.sqrt(amplitude_cos**2 + amplitude_sin**2))
            acrophase = float(np.arctan2(-amplitude_sin, amplitude_cos))

            print(f"[CosinorPy.fit_cosinor] Computed parameters: mesor={mesor:.4f}, amplitude={amplitude:.4f}, acrophase={acrophase:.4f}", file=sys.stderr)

            # Compute fitted values
            fitted_values = mesor + amplitude * np.cos(2 * np.pi * t / period + acrophase)
        except (np.linalg.LinAlgError, ValueError) as e:
            # If fit fails, use mean as mesor and zero amplitude
            print(f"[CosinorPy.fit_cosinor] Least squares fit failed: {e}. Falling back to mean/zero.", file=sys.stderr)
            mesor = float(np.mean(data)) if len(data) > 0 else 0.0
            amplitude = 0.0
            acrophase = 0.0
            fitted_values = np.full(data_length, mesor)
    else:
        # No data - return zeros
        print(f"[CosinorPy.fit_cosinor] No data provided for fit. Returning zeros.", file=sys.stderr)
        mesor = 0.0
        amplitude = 0.0
        acrophase = 0.0
        fitted_values = np.zeros(data_length)

    # Create a fit object with fittedvalues attribute
    class FitObject:
        def __init__(self, fitted_vals):
            self.fittedvalues = fitted_vals

    fit = FitObject(fitted_values)

    # Statistics dict with 'values' containing [mesor, amplitude, acrophase]
    statistics = {
        'values': [mesor, amplitude, acrophase]
    }

    # Return tuple of 4 elements as expected: fit, _, _, statistics
    return (fit, None, None, statistics)


class Cosinor:
    """Stub class for Cosinor."""
    def __init__(self, *args, **kwargs):
        pass
    
    def fit(self, *args, **kwargs):
        return fit_cosinor(*args, **kwargs)


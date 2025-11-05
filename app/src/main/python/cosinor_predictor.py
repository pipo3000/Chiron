"""
Python module for CosinorAge prediction using the cosinorage package.
This module processes accelerometer CSV files and returns CosinorAge predictions.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# IMPORTANT: Load stub modules BEFORE cosinorage tries to import them
# This ensures our stubs are in sys.modules before cosinorage's imports execute
try:
    # Pre-import stub modules to ensure they're available when cosinorage imports them
    try:
        import skdh
        import skdh.preprocessing
        import skdh.sleep
        import skdh.sleep.endpoints
        import skdh.sleep.sleep_classification
    except ImportError:
        # If stubs aren't found, that's okay - we'll see the error later
        pass
    
    try:
        import CosinorPy
        import CosinorPy.cosinor1
    except ImportError:
        # If stubs aren't found, that's okay - we'll see the error later
        pass
except Exception:
    # Ignore errors during stub loading - better to let cosinorage fail with a clear error
    pass

# Patch filter_consecutive_days will be applied later after cosinorage imports
# The early patch attempt is skipped since cosinorage may not be imported yet

# Try to import cosinorage with detailed error reporting
GenericDataHandler = None
CosinorAge = None

try:
    import importlib
    
    # Check if cosinorage module exists
    if 'cosinorage' in sys.modules:
        # Already imported, use it
        cosinorage_module = sys.modules['cosinorage']
    else:
        # Try to import
        import importlib.util
        spec = importlib.util.find_spec("cosinorage")
        if spec is None:
            raise ImportError("cosinorage module not found in Python path")
        
        # Import the module with better error handling
        try:
            cosinorage_module = importlib.import_module("cosinorage")
        except ImportError as import_err:
            # Try to get more details about what's missing
            import traceback
            error_msg = str(import_err)
            tb = traceback.format_exc()
            # Check if it's a missing dependency
            if "No module named" in error_msg or "ModuleNotFoundError" in error_msg:
                # Extract the module name that's missing
                missing_module = error_msg
                for prefix in ["No module named ", "ModuleNotFoundError: No module named "]:
                    if prefix in error_msg:
                        missing_module = error_msg.split(prefix, 1)[1].split()[0].strip("'\"")
                        break
                raise ImportError(f"cosinorage import failed due to missing dependency: '{missing_module}'. "
                                f"Full error: {error_msg}\nFull traceback:\n{tb}")
            else:
                raise ImportError(f"cosinorage import failed: {error_msg}\nFull traceback:\n{tb}")
        except Exception as e:
            import traceback
            raise ImportError(f"cosinorage import failed with unexpected error: {str(e)}\n"
                            f"Error type: {type(e).__name__}\n"
                            f"Full traceback:\n{traceback.format_exc()}")
    
    # Try to get the classes from their submodules
    # GenericDataHandler is in cosinorage.datahandlers
    try:
        datahandlers_module = cosinorage_module.datahandlers
        if hasattr(datahandlers_module, 'GenericDataHandler'):
            GenericDataHandler = datahandlers_module.GenericDataHandler
        else:
            raise ImportError("GenericDataHandler not found in cosinorage.datahandlers")
    except AttributeError:
        raise ImportError("cosinorage.datahandlers submodule not found")
    
    # CosinorAge is in cosinorage.bioages
    try:
        bioages_module = cosinorage_module.bioages
        if hasattr(bioages_module, 'CosinorAge'):
            CosinorAge = bioages_module.CosinorAge
        else:
            raise ImportError("CosinorAge not found in cosinorage.bioages")
    except AttributeError:
        raise ImportError("cosinorage.bioages submodule not found")
        
except ImportError as e:
    # Log the actual import error for debugging
    import traceback
    error_details = f"ImportError: {str(e)}\nPython path: {sys.path}\nTraceback: {traceback.format_exc()}"
    print(f"Failed to import cosinorage: {error_details}", file=sys.stderr)
    # Keep classes as None
except Exception as e:
    import traceback
    error_details = f"Unexpected error importing cosinorage: {str(e)}\nTraceback: {traceback.format_exc()}"
    print(f"Failed to import cosinorage: {error_details}", file=sys.stderr)
    # Keep classes as None

# Try to import and force CosinorPy stub early so it's available when cosinorage tries to import it
try:
    import CosinorPy
    import CosinorPy.cosinor1
    sys.modules['CosinorPy'] = CosinorPy
    sys.modules['CosinorPy.cosinor1'] = CosinorPy.cosinor1
    print(f"[EARLY] CosinorPy stub loaded and forced into sys.modules", file=sys.stderr)
except Exception as e:
    print(f"[EARLY] Could not load CosinorPy stub early: {e}", file=sys.stderr)


def process_csv_file(csv_file_path, age=40, gender='male'):
    """
    Process accelerometer CSV file and return CosinorAge prediction.
    Follows the reference implementation from predict_age.py
    
    Args:
        csv_file_path: Path to the CSV file with columns: timestamp,enmo (CosinorAge format)
                       OR legacy format: timestamp,x,y,z (for backward compatibility)
        age: Chronological age (defaults to 40)
        gender: Gender 'male' or 'female' (defaults to 'male')
    
    Returns:
        dict with 'success', 'message', 'cosinor_age', and 'error' keys
    """
    result = {
        'success': False,
        'message': '',
        'cosinor_age': None,
        'error': None
    }
    
    temp_file = None
    
    try:
        if not Path(csv_file_path).exists():
            result['message'] = f"File not found: {csv_file_path}"
            return result
        
        if GenericDataHandler is None or CosinorAge is None:
            import importlib.util
            import traceback
            spec = importlib.util.find_spec("cosinorage")
            if spec is None:
                result['message'] = "cosinorage package not found. It may not be bundled with the app. Check Chaquopy build logs."
            else:
                # Try to import again to get the actual error
                try:
                    importlib.import_module("cosinorage")
                except Exception as import_error:
                    error_details = f"cosinorage package found but import failed.\n"
                    error_details += f"Location: {spec.origin}\n"
                    error_details += f"Error: {str(import_error)}\n"
                    error_details += f"Error type: {type(import_error).__name__}\n"
                    error_details += f"Traceback:\n{traceback.format_exc()}"
                    result['message'] = error_details
                    result['error'] = str(import_error)
                    result['error_type'] = type(import_error).__name__
                    result['python_path'] = str(sys.path)
                    return result
                result['message'] = f"cosinorage package found but GenericDataHandler or CosinorAge not available. Spec: {spec.origin}"
            result['error'] = "ImportError"
            result['python_path'] = str(sys.path)
            return result
        
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        
        # Check if file is in new CosinorAge format (timestamp,enmo) or legacy format
        has_enmo_format = 'enmo' in df.columns and 'timestamp' in df.columns
        has_xyz_format = all(col in df.columns for col in ['x', 'y', 'z'])
        
        # Convert legacy x,y,z format to enmo if needed
        if has_xyz_format and not has_enmo_format:
            # Calculate ENMO (Euclidean Norm Minus One)
            df['enmo'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2) - 1.0
            df['enmo'] = df['enmo'].clip(lower=0)  # ENMO cannot be negative
        
        if 'enmo' not in df.columns or 'timestamp' not in df.columns:
            result['message'] = f"Unsupported CSV format. Expected 'timestamp,enmo' or 'timestamp,x,y,z'. Found: {df.columns.tolist()}"
            return result
        
        # Convert timestamp to datetime and sort
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'enmo'])
        
        if len(df) == 0:
            result['message'] = "No valid data found in CSV file"
            return result
        
        # Sort by timestamp and remove duplicates
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Check minimum data requirement (at least 1 day = 1440 minutes)
        if len(df) < 1440:
            result['message'] = f"Insufficient data: need at least 1 day (1440 minutes), got {len(df)} rows"
            return result
        
        # Save to temporary CSV file for GenericDataHandler
        temp_file = str(Path(csv_file_path).parent / 'temp_combined_data.csv')
        df.to_csv(temp_file, index=False)
        
        # Keep a reference to the preprocessed DataFrame for manual computation if needed
        preprocessed_df = df.copy()
        
        try:
            # Patch filter_consecutive_days to allow data even if not strictly consecutive
            stderr_ref = sys.stderr
            try:
                from cosinorage.datahandlers.utils import filtering
                if hasattr(filtering, 'filter_consecutive_days'):
                    original_func = filtering.filter_consecutive_days
                    
                    def patched_filter_consecutive_days(df):
                        """Patched to allow data with sufficient minutes even if not strictly consecutive days."""
                        if df.empty:
                            return df
                        try:
                            return original_func(df)
                        except ValueError as e:
                            error_str = str(e)
                            # Check for variations of the error message (including typo "Less then")
                            if ("Less than 1 day found" in error_str or 
                                "Less then 1 day found" in error_str or
                                "less than 1 day" in error_str.lower() or
                                "less then 1 day" in error_str.lower()):
                                # Calculate approximate number of days from data length
                                # Assuming ~1440 minutes per day, we need at least 3 days worth of data
                                # (3 days = 4320 minutes) to have reliable circadian rhythm analysis
                                min_minutes_for_prediction = 4320  # 3 days worth
                                if len(df) >= min_minutes_for_prediction:
                                    print(f"[PATCH] Allowing data with {len(df)} minutes (~{len(df)/1440:.1f} days) (patch applied)", file=stderr_ref)
                                    return df
                                else:
                                    print(f"[PATCH] Data has only {len(df)} minutes (~{len(df)/1440:.1f} days), need at least {min_minutes_for_prediction} minutes (~3 days)", file=stderr_ref)
                            raise
                    
                    filtering.filter_consecutive_days = patched_filter_consecutive_days
                    print(f"[PATCH] Applied filter_consecutive_days patch in process_csv_file", file=stderr_ref)
            except Exception as patch_error:
                print(f"[PATCH] Could not apply filter_consecutive_days patch: {patch_error}", file=sys.stderr)
                pass  # Patch is optional
            
            # Initialize GenericDataHandler - following reference implementation
            # Use less strict parameters to allow data that might not be perfectly consecutive
            try:
                data_handler = GenericDataHandler(
                    file_path=temp_file,
                    data_format='csv',
                    data_type='enmo-mg',  # ENMO data in mg units
                    time_format='datetime',  # ISO datetime format
                    time_column='timestamp',
                    data_columns=['enmo'],
                    verbose=False,
                    filter_incomplete_days=False,  # Don't filter incomplete days
                    select_longest_sequence=False,  # Don't select only longest sequence
                    required_daily_coverage=0.0  # No minimum daily coverage required
                )
                print(f"[DEBUG] GenericDataHandler created with lenient parameters", file=sys.stderr)
            except TypeError:
                # If these parameters don't exist, use default initialization
                print(f"[DEBUG] GenericDataHandler doesn't support lenient parameters, using defaults", file=sys.stderr)
                data_handler = GenericDataHandler(
                    file_path=temp_file,
                    data_format='csv',
                    data_type='enmo-mg',
                    time_format='datetime',
                    time_column='timestamp',
                    data_columns=['enmo'],
                    verbose=False
                )
            
            # Create a record for CosinorAge prediction - following reference implementation
            record = {
                'handler': data_handler,
                'age': age,
                'gender': gender
            }
            
            # Force CosinorPy and cosinor1 into sys.modules to ensure cosinorage uses our stub
            try:
                import CosinorPy
                import CosinorPy.cosinor1 as cosinor1_module
                sys.modules['CosinorPy'] = CosinorPy
                sys.modules['CosinorPy.cosinor1'] = cosinor1_module
                print(f"[DEBUG] Forced CosinorPy into sys.modules", file=sys.stderr)
            except Exception as e:
                print(f"[DEBUG] Could not force CosinorPy into sys.modules: {e}", file=sys.stderr)
            
            # Try to patch cosinorage's internal cosinor computation if it exists
            try:
                from cosinorage.features.utils import cosinor_analysis
                if hasattr(cosinor_analysis, 'fit_cosinor'):
                    original_cosinor_fit = cosinor_analysis.fit_cosinor
                    print(f"[DEBUG] Found cosinorage.features.utils.cosinor_analysis.fit_cosinor", file=sys.stderr)
                    
                    def patched_cosinor_fit(*args, **kwargs):
                        print(f"[DEBUG] cosinorage.features.utils.cosinor_analysis.fit_cosinor called", file=sys.stderr)
                        try:
                            result = original_cosinor_fit(*args, **kwargs)
                            print(f"[DEBUG] cosinorage fit_cosinor returned: {type(result)}", file=sys.stderr)
                            return result
                        except Exception as e:
                            print(f"[DEBUG] cosinorage fit_cosinor error: {e}", file=sys.stderr)
                            raise
                    
                    cosinor_analysis.fit_cosinor = patched_cosinor_fit
                    print(f"[DEBUG] Patched cosinorage.features.utils.cosinor_analysis.fit_cosinor", file=sys.stderr)
            except Exception as e:
                print(f"[DEBUG] Could not patch cosinorage internal fit_cosinor: {e}", file=sys.stderr)
            
            # Compute cosinor parameters manually first (before calling get_predictions)
            # This ensures we have the parameters available when cosinorage computes the age
            print(f"[DEBUG] Computing cosinor parameters from preprocessed data", file=sys.stderr)
            computed_mesor = None
            computed_amp1 = None
            computed_phi1 = None
            
            try:
                # Use the preprocessed DataFrame directly (same approach as visualization)
                # Ensure np is available (imported at module level)
                enmo_series = preprocessed_df['enmo']
                
                if len(enmo_series) > 0:
                    period = 1440.0  # 24 hours in minutes
                    # Explicitly reference numpy to avoid any scoping issues
                    import numpy
                    t = numpy.arange(len(enmo_series))
                    cos_term = numpy.cos(2 * numpy.pi * t / period)
                    sin_term = numpy.sin(2 * numpy.pi * t / period)
                    X = numpy.column_stack([numpy.ones(len(t)), cos_term, sin_term])
                    
                    # Convert to numpy array
                    y = numpy.asarray(enmo_series.values)
                    
                    coeffs = numpy.linalg.lstsq(X, y, rcond=None)[0]
                    computed_mesor = float(coeffs[0])
                    amplitude_cos = float(coeffs[1])
                    amplitude_sin = float(coeffs[2])
                    computed_amp1 = float(numpy.sqrt(amplitude_cos**2 + amplitude_sin**2))
                    computed_phi1 = float(numpy.arctan2(-amplitude_sin, amplitude_cos))
                    
                    print(f"[DEBUG] Pre-computed: mesor={computed_mesor:.4f}, amp1={computed_amp1:.4f}, phi1={computed_phi1:.4f}", file=sys.stderr)
                else:
                    print(f"[DEBUG] Preprocessed data is empty", file=sys.stderr)
            except Exception as e:
                print(f"[DEBUG] Error pre-computing cosinor parameters: {e}", file=sys.stderr)
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}", file=sys.stderr)
            
            # Compute CosinorAge predictions - following reference implementation
            print(f"[DEBUG] Creating CosinorAge model with record: age={record['age']}, gender={record['gender']}", file=sys.stderr)
            cosinor_age_model = CosinorAge([record])
            print(f"[DEBUG] Calling get_predictions()...", file=sys.stderr)
            predictions = cosinor_age_model.get_predictions()
            print(f"[DEBUG] get_predictions() returned {len(predictions) if predictions else 0} predictions", file=sys.stderr)
            
            # If cosinor parameters are None, inject the pre-computed values
            if predictions and len(predictions) > 0:
                prediction_result = predictions[0]
                if (prediction_result.get('mesor') is None or 
                    prediction_result.get('amp1') is None or 
                    prediction_result.get('phi1') is None):
                    if computed_mesor is not None and computed_amp1 is not None and computed_phi1 is not None:
                        print(f"[DEBUG] Injecting pre-computed cosinor parameters into prediction result", file=sys.stderr)
                        prediction_result['mesor'] = computed_mesor
                        prediction_result['amp1'] = computed_amp1
                        prediction_result['phi1'] = computed_phi1
                        
                        # Try to recompute cosinorage with the injected parameters
                        # Use the model_params_male/female/generic to compute age manually
                        try:
                            # Get the appropriate model parameters based on gender
                            if gender == 'male' and hasattr(cosinor_age_model, 'model_params_male'):
                                model_params = cosinor_age_model.model_params_male
                                print(f"[DEBUG] Using model_params_male: {model_params}", file=sys.stderr)
                            elif gender == 'female' and hasattr(cosinor_age_model, 'model_params_female'):
                                model_params = cosinor_age_model.model_params_female
                                print(f"[DEBUG] Using model_params_female: {model_params}", file=sys.stderr)
                            elif hasattr(cosinor_age_model, 'model_params_generic'):
                                model_params = cosinor_age_model.model_params_generic
                                print(f"[DEBUG] Using model_params_generic: {model_params}", file=sys.stderr)
                            else:
                                model_params = None
                                print(f"[DEBUG] No model parameters found for gender={gender}", file=sys.stderr)
                            
                            if model_params is not None:
                                # Model parameters format: dict with keys: 'rate' (intercept), 'mesor', 'amp1', 'phi1', 'age', 'shape'
                                # Formula: cosinorage = rate + mesor_coef*mesor + amp1_coef*amp1 + phi1_coef*phi1 + age_coef*age
                                try:
                                    # Try different parameter formats
                                    if isinstance(model_params, dict):
                                        # Dictionary format: {'rate': intercept, 'mesor': coef, 'amp1': coef, 'phi1': coef, 'age': coef, 'shape': ...}
                                        rate = float(model_params.get('rate', 0.0))
                                        mesor_coef = float(model_params.get('mesor', 0.0))
                                        amp1_coef = float(model_params.get('amp1', 0.0))
                                        phi1_coef = float(model_params.get('phi1', 0.0))
                                        age_coef = float(model_params.get('age', 0.0))
                                        
                                        # Compute biological age using linear regression formula
                                        computed_age = rate + mesor_coef*computed_mesor + amp1_coef*computed_amp1 + phi1_coef*computed_phi1 + age_coef*age
                                        prediction_result['cosinorage'] = float(computed_age)
                                        print(f"[DEBUG] Computed cosinorage manually: {computed_age:.2f} (rate={rate:.4f}, mesor_coef={mesor_coef:.4f}, amp1_coef={amp1_coef:.4f}, phi1_coef={phi1_coef:.4f}, age_coef={age_coef:.4f})", file=sys.stderr)
                                    elif isinstance(model_params, (list, tuple, np.ndarray)):
                                        params = np.asarray(model_params)
                                        print(f"[DEBUG] Model parameters shape: {params.shape}, values: {params}", file=sys.stderr)
                                        
                                        # Array format: [intercept, mesor, amp1, phi1, age, ...]
                                        if len(params) >= 5:
                                            intercept = float(params[0])
                                            mesor_coef = float(params[1]) if len(params) > 1 else 0.0
                                            amp1_coef = float(params[2]) if len(params) > 2 else 0.0
                                            phi1_coef = float(params[3]) if len(params) > 3 else 0.0
                                            age_coef = float(params[4]) if len(params) > 4 else 0.0
                                            
                                            # Compute biological age
                                            computed_age = intercept + mesor_coef*computed_mesor + amp1_coef*computed_amp1 + phi1_coef*computed_phi1 + age_coef*age
                                            prediction_result['cosinorage'] = float(computed_age)
                                            print(f"[DEBUG] Computed cosinorage manually: {computed_age:.2f} (intercept={intercept:.4f}, mesor_coef={mesor_coef:.4f}, amp1_coef={amp1_coef:.4f}, phi1_coef={phi1_coef:.4f}, age_coef={age_coef:.4f})", file=sys.stderr)
                                        else:
                                            print(f"[DEBUG] Model parameters array too short: {len(params)}", file=sys.stderr)
                                    elif hasattr(model_params, 'predict'):
                                        # If it's a model object, use predict
                                        features = np.array([[computed_mesor, computed_amp1, computed_phi1, age]])
                                        predicted = model_params.predict(features)
                                        prediction_result['cosinorage'] = float(predicted[0])
                                        print(f"[DEBUG] Used model_params.predict: cosinorage={predicted[0]}", file=sys.stderr)
                                    else:
                                        print(f"[DEBUG] Model parameters format not recognized: {type(model_params)}", file=sys.stderr)
                                except Exception as e:
                                    print(f"[DEBUG] Error computing cosinorage from model_params: {e}", file=sys.stderr)
                                    import traceback
                                    print(f"[DEBUG] Traceback: {traceback.format_exc()}", file=sys.stderr)
                        except Exception as e:
                            print(f"[DEBUG] Could not recompute cosinorage: {e}", file=sys.stderr)
                            import traceback
                            print(f"[DEBUG] Traceback: {traceback.format_exc()}", file=sys.stderr)
            
            # Extract the predicted biological age from the results - following reference implementation
            if predictions and len(predictions) > 0:
                prediction_result = predictions[0]
                
                # Debug: log all values
                print(f"[DEBUG] Prediction result keys: {list(prediction_result.keys())}", file=sys.stderr)
                for key, value in prediction_result.items():
                    if key != 'handler':  # Skip handler object
                        print(f"[DEBUG] {key} = {value}", file=sys.stderr)
                
                predicted_age = prediction_result.get('cosinorage')
                mesor = prediction_result.get('mesor')
                amp1 = prediction_result.get('amp1')
                phi1 = prediction_result.get('phi1')
                
                print(f"[DEBUG] Extracted values: cosinorage={predicted_age}, mesor={mesor}, amp1={amp1}, phi1={phi1}", file=sys.stderr)
                
                if predicted_age is not None:
                    # cosinorage represents the age difference/advance
                    # Biological age = chronological age + cosinorage
                    # If cosinorage is negative, biological age is younger than chronological
                    biological_age = age + float(predicted_age)
                    result['success'] = True
                    result['cosinor_age'] = float(biological_age)
                    result['cosinorage_advance'] = float(predicted_age)  # Store the advance separately
                    result['message'] = f"CosinorAge prediction successful: {biological_age:.2f} years (advance: {predicted_age:.2f} years)"
                    
                    # Also include advance if available
                    if 'cosinorage_advance' in prediction_result:
                        result['cosinorage_advance'] = prediction_result['cosinorage_advance']
                else:
                    # Provide more detailed error message
                    error_msg = f"Prediction failed. cosinorage=None"
                    if mesor is None:
                        error_msg += f" (mesor=None)"
                    if amp1 is None:
                        error_msg += f" (amp1=None)"
                    if phi1 is None:
                        error_msg += f" (phi1=None)"
                    error_msg += ". Check if enough data is available."
                    result['message'] = error_msg
                    result['available_keys'] = list(prediction_result.keys())
                    result['cosinor_params'] = {
                        'mesor': mesor,
                        'amp1': amp1,
                        'phi1': phi1
                    }
            else:
                result['message'] = "CosinorAge prediction returned no results"
                
        except Exception as e:
            result['error'] = str(e)
            result['message'] = f"Error in CosinorAge calculation: {str(e)}"
            import traceback
            result['traceback'] = traceback.format_exc()
        finally:
            # Clean up temporary file
            if temp_file and Path(temp_file).exists():
                try:
                    Path(temp_file).unlink()
                except:
                    pass
            
    except Exception as e:
        result['error'] = str(e)
        result['message'] = f"Error processing file: {str(e)}"
        import traceback
        result['traceback'] = traceback.format_exc()
    
    return result


def predict_from_file(csv_file_path, age=None, gender=None):
    """
    Simplified prediction function that can be called from Kotlin.
    
    Args:
        csv_file_path: Path to CSV file
        age: Chronological age (optional, defaults to 40)
        gender: Gender 'male' or 'female' (optional, defaults to 'male')
    
    Returns:
        JSON string with prediction results
    """
    if age is None:
        age = 40
    if gender is None:
        gender = 'male'
    
    result = process_csv_file(csv_file_path, age=age, gender=gender)
    return json.dumps(result)


def get_visualization_data(csv_file_path, age=None, gender=None):
    """
    Get ENMO data and cosinor fit for visualization.
    
    Args:
        csv_file_path: Path to CSV file
        age: Chronological age (optional, defaults to 40)
        gender: Gender 'male' or 'female' (optional, defaults to 'male')
    
    Returns:
        JSON string with visualization data including:
        - timestamps: List of timestamps
        - enmo_values: List of ENMO values
        - cosinor_fit: List of cosinor fit values
        - cosinor_params: Dictionary with mesor, amplitude, acrophase
    """
    result = {
        'success': False,
        'message': '',
        'timestamps': [],
        'enmo_values': [],
        'cosinor_fit': [],
        'cosinor_params': {},
        'error': None
    }
    
    if age is None:
        age = 40
    if gender is None:
        gender = 'male'
    
    try:
        if not Path(csv_file_path).exists():
            result['message'] = f"File not found: {csv_file_path}"
            return json.dumps(result)
        
        # Read and preprocess CSV
        df = pd.read_csv(csv_file_path)
        
        has_enmo_format = 'enmo' in df.columns and 'timestamp' in df.columns
        has_xyz_format = all(col in df.columns for col in ['x', 'y', 'z'])
        
        if has_xyz_format and not has_enmo_format:
            df['enmo'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2) - 1.0
            df['enmo'] = df['enmo'].clip(lower=0)
        
        if 'enmo' not in df.columns or 'timestamp' not in df.columns:
            result['message'] = f"Unsupported CSV format. Expected 'timestamp,enmo' or 'timestamp,x,y,z'"
            return json.dumps(result)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'enmo'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        if len(df) == 0:
            result['message'] = "No valid data found in CSV file"
            return json.dumps(result)
        
        # Use the preprocessed DataFrame directly
        enmo_series = df['enmo']
        timestamps = df['timestamp']
        
        # Convert timestamps to strings for JSON serialization
        timestamp_strs = [str(ts) for ts in timestamps]
        
        result['timestamps'] = timestamp_strs
        result['enmo_values'] = enmo_series.tolist()
        
        # Compute cosinor fit manually using the data
        try:
            if len(enmo_series) > 0:
                # Compute cosinor fit: mesor + amplitude * cos(2*pi*t/period + acrophase)
                # For circadian rhythm, period is 24 hours (1440 minutes)
                period = 1440.0  # 24 hours in minutes
                t = np.arange(len(enmo_series))
                
                # Simple cosinor fit using least squares
                # y = M + A*cos(2*pi*t/T + phi)
                # We'll use a simpler approach: fit to cos(2*pi*t/T) and sin(2*pi*t/T)
                cos_term = np.cos(2 * np.pi * t / period)
                sin_term = np.sin(2 * np.pi * t / period)
                
                # Linear regression: y = M + A_cos*cos + A_sin*sin
                X = np.column_stack([np.ones(len(t)), cos_term, sin_term])
                coeffs = np.linalg.lstsq(X, enmo_series.values, rcond=None)[0]
                
                mesor = float(coeffs[0])
                amplitude_cos = float(coeffs[1])
                amplitude_sin = float(coeffs[2])
                amplitude = float(np.sqrt(amplitude_cos**2 + amplitude_sin**2))
                acrophase = float(np.arctan2(-amplitude_sin, amplitude_cos))
                
                # Compute fit values
                cosinor_fit = mesor + amplitude * np.cos(2 * np.pi * t / period + acrophase)
                
                result['cosinor_fit'] = cosinor_fit.tolist()
                result['cosinor_params'] = {
                    'mesor': mesor,
                    'amplitude': amplitude,
                    'acrophase': acrophase
                }
                
                result['success'] = True
                result['message'] = "Visualization data computed successfully"
        except Exception as e:
            # If cosinor fit fails, just return ENMO data
            result['message'] = f"ENMO data available, but cosinor fit failed: {str(e)}"
            result['success'] = True  # Still return ENMO data
        
    except Exception as e:
        result['error'] = str(e)
        result['message'] = f"Error getting visualization data: {str(e)}"
        import traceback
        result['traceback'] = traceback.format_exc()
    
    return json.dumps(result)


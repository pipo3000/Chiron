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
            spec = importlib.util.find_spec("cosinorage")
            if spec is None:
                result['message'] = "cosinorage package not found. It may not be bundled with the app. Check Chaquopy build logs."
            else:
                result['message'] = f"cosinorage package found but import failed. Check Python dependencies. Spec: {spec.origin}"
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
        
        try:
            # Patch filter_consecutive_days to allow single-day data if needed
            stderr_ref = sys.stderr
            try:
                from cosinorage.datahandlers.utils import filtering
                if hasattr(filtering, 'filter_consecutive_days'):
                    original_func = filtering.filter_consecutive_days
                    
                    def patched_filter_consecutive_days(df):
                        """Patched to allow single-day data (>= 1000 minutes)."""
                        if df.empty:
                            return df
                        try:
                            return original_func(df)
                        except ValueError as e:
                            if "Less than 1 day found" in str(e) and len(df) >= 1000:
                                print(f"Allowing single-day data: {len(df)} minutes", file=stderr_ref)
                                return df
                            raise
                    
                    filtering.filter_consecutive_days = patched_filter_consecutive_days
            except Exception:
                pass  # Patch is optional
            
            # Initialize GenericDataHandler - following reference implementation
            data_handler = GenericDataHandler(
                file_path=temp_file,
                data_format='csv',
                data_type='enmo-mg',  # ENMO data in mg units
                time_format='datetime',  # ISO datetime format
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
            
            # Compute CosinorAge predictions - following reference implementation
            cosinor_age_model = CosinorAge([record])
            predictions = cosinor_age_model.get_predictions()
            
            # Extract the predicted biological age from the results - following reference implementation
            if predictions and len(predictions) > 0:
                prediction_result = predictions[0]
                predicted_age = prediction_result.get('cosinorage')
                
                if predicted_age is not None:
                    result['success'] = True
                    result['cosinor_age'] = float(predicted_age)
                    result['message'] = f"CosinorAge prediction successful: {predicted_age:.2f} years"
                    
                    # Also include advance if available
                    if 'cosinorage_advance' in prediction_result:
                        result['cosinorage_advance'] = prediction_result['cosinorage_advance']
                else:
                    result['message'] = "Prediction failed. Check if enough data is available."
                    result['available_keys'] = list(prediction_result.keys())
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


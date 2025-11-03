# CosinorAge File Format Requirements

## Input File Format (What Your App Currently Produces)

The app currently writes files in this format:

**CSV File Format:**
```
timestamp,x,y,z,magnitude
1699123456789,0.5,-0.3,0.8,0.97
1699123457123,0.6,-0.2,0.7,0.94
...
```

**Column Requirements:**
- `timestamp`: Unix timestamp in **milliseconds** (Long integer)
- `x`: X-axis acceleration (Float)
- `y`: Y-axis acceleration (Float)  
- `z`: Z-axis acceleration (Float)
- `magnitude`: Optional - calculated as √(x² + y² + z²) but not used by CosinorAge

## What CosinorAge Actually Needs

The Python script processes your files and converts them to the format CosinorAge expects:

### 1. Initial Processing (Your CSV → Internal Format)

**Required columns in input CSV:**
- ✅ `timestamp` (milliseconds, will be converted to datetime)
- ✅ `x`, `y`, `z` (acceleration values)
- ⚠️ `magnitude` (ignored, but doesn't hurt to include)

**The script automatically:**
1. Calculates **ENMO** (Euclidean Norm Minus One): `ENMO = max(0, √(x² + y² + z²) - 1)`
2. Converts timestamps from milliseconds to pandas datetime
3. Resamples data to **minute-level resolution** (1 data point per minute)

### 2. Final Format for CosinorAge

**Temporary file created (timestamp,enmo):**
```
timestamp,enmo
2023-11-04 00:00:00,0.0234
2023-11-04 00:01:00,0.0156
2023-11-04 00:02:00,0.0189
...
```

**GenericDataHandler expects:**
- `time_column`: `'timestamp'` (datetime format)
- `data_columns`: `['enmo']` (ENMO values)
- `data_type`: `'enmo-mg'` (ENMO in milligravity units)

## Data Requirements

### Minimum Data Volume
- **At least 1 day of data** = **1440 minutes** of minute-level data
- For raw accelerometer data, this means you need sufficient data points to produce 1440 minute-level samples

### Data Quality
- Timestamps must be valid (non-null)
- Acceleration values should be reasonable (typically -20 to +20 g)
- ENMO values are automatically clipped to be non-negative

### Time Resolution
- **Raw data**: Can be at any frequency (your app uses `SENSOR_DELAY_NORMAL`)
- **Processed data**: Automatically resampled to **1-minute intervals** (mean value per minute)

## Current Implementation Status

✅ **Your app's format is CORRECT and compatible!**

Your `AccelerometerDataService` writes:
```csv
timestamp,x,y,z,magnitude
```

The Python script:
1. ✅ Reads this format correctly
2. ✅ Extracts `timestamp`, `x`, `y`, `z` (ignores `magnitude`)
3. ✅ Calculates ENMO automatically
4. ✅ Converts to minute-level resolution
5. ✅ Creates the required format for CosinorAge

## Example File Structure

**Your app writes:**
```csv
timestamp,x,y,z,magnitude
1699123456789,0.523,-0.312,0.823,1.023
1699123456890,0.601,-0.234,0.712,0.945
1699123456991,0.567,-0.289,0.789,0.987
...
```

**After processing by Python script (internal, temporary):**
```csv
timestamp,enmo
2023-11-04 00:00:00,0.023
2023-11-04 00:01:00,0.015
2023-11-04 00:02:00,0.018
...
```

**What CosinorAge receives:**
- Minute-level ENMO values
- Datetime-indexed timestamps
- At least 1 day (1440 minutes) of continuous data

## Summary

✅ **Keep your current format** - it's perfect!
- CSV with `timestamp,x,y,z,magnitude` columns
- Timestamps in milliseconds
- The Python script handles all conversions automatically

⚠️ **Make sure you have enough data:**
- Record at least 1 full day before running prediction
- The script will warn if there's insufficient data (< 1440 minutes)


# Chiron

Chiron is an Android application that predicts biological age from continuous accelerometer data using circadian rhythm analysis. The app uses the `cosinorage` Python package to analyze activity patterns and provide insights into your biological age based on movement data.

## Overview

Chiron continuously records accelerometer data from your device and uses advanced cosinor analysis to extract circadian rhythm features. These features are then used to predict biological age, providing insights into how your body's internal clock and activity patterns relate to aging.

## Key Features

- **Continuous Background Recording**: Automatically records accelerometer data in the background, even when the app is closed
- **Daily Biological Age Prediction**: Uses up to 14 days of historical accelerometer data to predict biological age
- **Automatic Daily Updates**: Predictions automatically refresh each day as new data becomes available
- **Minute-Level Aggregation**: Aggregates accelerometer data into minute-level ENMO (Euclidean Norm Minus One) values for analysis
- **Python Integration**: Uses Chaquopy to integrate Python scientific computing libraries (pandas, numpy, scipy, matplotlib) and the `cosinorage` package

## How It Works

1. **Data Collection**: The app continuously records accelerometer data (x, y, z values) using a foreground service
2. **Data Processing**: Raw accelerometer data is converted to ENMO (Euclidean Norm Minus One) values and aggregated by minute
3. **Data Storage**: Daily CSV files are stored in `Downloads/Chiron/` with format: `accelerometer_YYYYMMDD.csv`
4. **Prediction**: The app uses the last 14 days of data (excluding today) to predict biological age using:
   - Circadian rhythm analysis (cosinor fitting)
   - Activity pattern extraction
   - Sleep/wake detection
   - Wearable feature computation
5. **Display**: The predicted biological age is displayed in the app interface

## Technical Details

### Data Format

CSV files contain minute-level aggregated data:
```
timestamp,enmo
2025-10-30T00:00:00,0.123
2025-10-30T00:01:00,0.456
...
```

- **timestamp**: ISO 8601 format (YYYY-MM-DDTHH:mm:ss)
- **enmo**: Euclidean Norm Minus One, calculated as `max(0, sqrt(x² + y² + z²) - 1.0)`

### Python Dependencies

The app uses Chaquopy to embed Python 3.9 and the following packages:
- `cosinorage==1.0.5` - Biological age prediction from accelerometer data
- `claid` - Data processing utilities
- `seaborn` - Visualization (dependency)
- `pandas`, `numpy`, `scipy`, `matplotlib` - Pre-installed by Chaquopy

### Stub Modules

Some dependencies (`skdh` - scikit-digital-health, `CosinorPy`) cannot be built for Android. The app includes minimal stub modules to allow `cosinorage` to load, though some advanced features may not be available.

### Prediction Algorithm

The prediction uses the `cosinorage` package which:
1. Loads accelerometer data using `GenericDataHandler`
2. Extracts wearable features (circadian rhythm parameters, activity metrics, sleep metrics)
3. Computes cosinor features (mesor, amplitude, acrophase)
4. Predicts biological age using the `CosinorAge` model

## Requirements

- Android 7.0 (API level 24) or higher
- Accelerometer sensor
- Storage permission for saving data files
- Notification permission (Android 13+) for foreground service

## Build Instructions

1. Clone the repository
2. Open the project in Android Studio
3. Ensure you have the Android SDK and Gradle configured
4. Build the project using Gradle:
   ```bash
   ./gradlew assembleDebug
   ```

The build process will:
- Download Python 3.9 via Chaquopy
- Install Python packages via pip
- Bundle Python code and dependencies with the APK

## Project Structure

```
app/
├── src/main/
│   ├── java/com/example/chiron/
│   │   ├── MainActivity.kt          # Main UI and prediction display
│   │   ├── AccelerometerDataService.kt  # Background recording service
│   │   └── CosinorAgePredictor.kt   # Python integration and file management
│   └── python/
│       ├── cosinor_predictor.py     # Main prediction logic
│       ├── skdh/                    # Stub modules for skdh
│       └── CosinorPy/               # Stub modules for CosinorPy
```

## Data Storage

- **Primary Location**: `Downloads/Chiron/accelerometer_*.csv`
- **Fallback Location**: App's external files directory
- **File Naming**: `accelerometer_YYYYMMDD.csv` (one file per day)
- **Retention**: Files older than 10 days are automatically cleaned up

## Notes

- The app requires the device accelerometer to be active
- A foreground service keeps the app recording in the background
- Predictions use data from previous days (excluding today) to ensure complete daily data
- The app generates synthetic test data on first launch if no real data exists

## License

This project is licensed under a non-commercial license. All rights reserved.

**Non-Commercial Use Only**: This software and associated documentation files (the "Software") are provided for personal, non-commercial use only. You may not use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software for commercial purposes without express written permission from the author.

**Attribution**: If you use this Software in any non-commercial work, please provide attribution to the original author.

## Author

**Filipe Barata**

Developer of Chiron - Biological Age Prediction from Accelerometer Data

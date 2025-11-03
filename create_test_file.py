#!/usr/bin/env python3
"""
Script to create a synthetic test file for CosinorAge prediction.
Run this script and transfer the generated file to your Android device's Downloads/Chiron directory.
Or use ADB to push it: adb push accelerometer_YYYYMMDD.csv /sdcard/Download/Chiron/
"""

import datetime
import random
import os

def create_synthetic_file(days_ago=1):
    """Create a synthetic accelerometer file with ENMO data for testing."""
    
    # Get the date
    target_date = datetime.datetime.now() - datetime.timedelta(days=days_ago)
    date_str = target_date.strftime('%Y%m%d')
    
    # Calculate start of day at 00:00:00
    start_time = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Generate 1440 minutes (1 full day) of data
    data = []
    for minute in range(1440):
        timestamp = start_time + datetime.timedelta(minutes=minute)
        timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Generate realistic ENMO values
        # Lower at night (sleep), higher during day (activity)
        hour = timestamp.hour
        
        # Base ENMO varies by time of day
        if 2 <= hour <= 6:  # Deep sleep hours (2 AM - 6 AM)
            base_enmo = 0.01 + random.uniform(0, 0.02)
        elif 7 <= hour <= 9 or 17 <= hour <= 22:  # Active hours (morning/evening)
            base_enmo = 0.05 + random.uniform(0, 0.15)
        else:  # Other hours
            base_enmo = 0.02 + random.uniform(0, 0.08)
        
        # Add some noise for realism
        enmo = base_enmo + random.uniform(-0.01, 0.01)
        enmo = max(0.0, enmo)  # ENMO can't be negative
        
        data.append(f'{timestamp_str},{enmo:.6f}')
    
    # Create filename
    filename = f'accelerometer_{date_str}.csv'
    
    # Write to file
    with open(filename, 'w') as f:
        f.write('timestamp,enmo\n')
        f.write('\n'.join(data))
    
    print(f'✓ Created: {filename}')
    print(f'  Date: {target_date.strftime("%Y-%m-%d")}')
    print(f'  Total minutes: {len(data)} (1 full day)')
    print(f'  File size: {os.path.getsize(filename)} bytes')
    print(f'\n  To transfer to Android device:')
    print(f'    adb push {filename} /sdcard/Download/Chiron/')
    
    return filename

if __name__ == '__main__':
    import sys
    days_ago = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    create_synthetic_file(days_ago)


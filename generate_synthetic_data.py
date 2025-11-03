#!/usr/bin/env python3
"""
Script to generate synthetic accelerometer data for the last two days.
Generates files in the format: timestamp,enmo (ISO 8601 format)
Run this script and transfer the generated files to your Android device's Downloads/Chiron directory.
"""

import datetime
import random
import os
from pathlib import Path

def generate_realistic_enmo(hour, minute):
    """
    Generate realistic ENMO values based on time of day.
    Lower at night (sleep), higher during day (activity).
    """
    # Base ENMO varies by time of day with circadian rhythm
    # Typical sleep hours: 22:00 - 06:00 (10 PM - 6 AM)
    # Active hours: 07:00 - 09:00 (morning) and 17:00 - 21:00 (evening)
    
    if 22 <= hour or hour < 6:  # Sleep hours (10 PM - 6 AM)
        base_enmo = 0.005 + random.uniform(0, 0.015)
    elif 6 <= hour < 9:  # Morning wake-up (6 AM - 9 AM)
        base_enmo = 0.03 + random.uniform(0, 0.08)
    elif 9 <= hour < 12:  # Morning activity (9 AM - 12 PM)
        base_enmo = 0.05 + random.uniform(0, 0.12)
    elif 12 <= hour < 14:  # Midday (12 PM - 2 PM)
        base_enmo = 0.04 + random.uniform(0, 0.10)
    elif 14 <= hour < 17:  # Afternoon (2 PM - 5 PM)
        base_enmo = 0.06 + random.uniform(0, 0.15)
    elif 17 <= hour < 21:  # Evening activity (5 PM - 9 PM)
        base_enmo = 0.08 + random.uniform(0, 0.18)
    else:  # 21:00 - 22:00 (9 PM - 10 PM) - winding down
        base_enmo = 0.02 + random.uniform(0, 0.06)
    
    # Add some minute-level variation for realism
    minute_variation = random.uniform(-0.01, 0.01)
    enmo = base_enmo + minute_variation
    enmo = max(0.0, enmo)  # ENMO can't be negative
    
    return round(enmo, 6)

def create_synthetic_file(target_date, output_dir="."):
    """
    Create a synthetic accelerometer file for a specific date.
    
    Args:
        target_date: datetime.date object for the date to generate
        output_dir: Directory to write the file to
    
    Returns:
        Path to the created file
    """
    date_str = target_date.strftime('%Y%m%d')
    
    # Calculate start of day at 00:00:00
    start_time = datetime.datetime.combine(target_date, datetime.time(0, 0, 0))
    
    # Generate 1440 minutes (1 full day) of data
    data_lines = []
    for minute in range(1440):
        timestamp = start_time + datetime.timedelta(minutes=minute)
        timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Generate realistic ENMO value based on time of day
        enmo = generate_realistic_enmo(timestamp.hour, timestamp.minute)
        
        data_lines.append(f'{timestamp_str},{enmo}')
    
    # Create filename
    filename = f'accelerometer_{date_str}.csv'
    filepath = os.path.join(output_dir, filename)
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write('timestamp,enmo\n')
        f.write('\n'.join(data_lines))
    
    print(f'✓ Created: {filename}')
    print(f'  Date: {target_date.strftime("%Y-%m-%d")}')
    print(f'  Total minutes: {len(data_lines)} (1 full day)')
    print(f'  File size: {os.path.getsize(filepath)} bytes')
    
    return filepath

def main():
    """Generate synthetic data for the last week (7 days)."""
    print("Generating synthetic accelerometer data for the last 7 days...")
    print("=" * 60)
    
    # Create Downloads/Chiron directory if it doesn't exist
    download_dir = os.path.expanduser("~/Downloads/Chiron")
    os.makedirs(download_dir, exist_ok=True)
    print(f"\nOutput directory: {download_dir}")
    
    today = datetime.date.today()
    
    generated_files = []
    
    # Generate for the last 7 days (6 days ago through yesterday)
    for days_ago in range(6, -1, -1):  # 6, 5, 4, 3, 2, 1, 0 (0 = today, but we'll skip today)
        if days_ago == 0:
            continue  # Skip today, only generate past days
        target_date = today - datetime.timedelta(days=days_ago)
        print(f"\nGenerating data for {target_date.strftime('%Y-%m-%d')} ({days_ago} day(s) ago):")
        filepath = create_synthetic_file(target_date, output_dir=download_dir)
        generated_files.append(filepath)
    
    print("\n" + "=" * 60)
    print(f"✓ All {len(generated_files)} files generated successfully!")
    print(f"\nFiles saved to: {download_dir}")
    print("\nGenerated files:")
    for filepath in generated_files:
        filename = os.path.basename(filepath)
        print(f"  - {filename}")
    
    print("\n" + "=" * 60)
    print("\nTo transfer these files to your Android device:")
    print(f"1. Using ADB: adb push {download_dir}/accelerometer_*.csv /sdcard/Download/Chiron/")
    print(f"2. Or manually copy files from: {download_dir}")
    print("\nFiles are in the correct format (timestamp,enmo) for cosinorage.")

if __name__ == '__main__':
    main()


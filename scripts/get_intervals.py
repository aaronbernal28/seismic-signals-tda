"""
Script to generate time intervals for seismic events and non-events.
Reads raw event data and creates signal_intervals.csv with:
- Event intervals (label=1): Random windows around each event
- Non-event intervals (label=0): Random 150s windows where no events occur
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
from obspy import read_events

# Add parent directory to path to allow imports from config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, START_TIME, END_TIME


def extract_event_data(catalog):
    """
    Extract event information from ObsPy catalog.
    
    Parameters:
    -----------
    catalog : obspy.core.event.Catalog
        The event catalog to process
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, lat, long, mag, id
    """
    events_data = []
    
    for event in catalog:
        # Get preferred origin or first origin
        origin = event.preferred_origin() or event.origins[0]
        
        # Get preferred magnitude or first magnitude
        magnitude = event.preferred_magnitude() or event.magnitudes[0] if event.magnitudes else None
        
        # Extract event ID
        event_id = str(event.resource_id).split('/')[-1] if event.resource_id else None
        
        # Build event dictionary
        event_dict = {
            'date': origin.time.datetime,
            'lat': origin.latitude,
            'long': origin.longitude,
            'mag': magnitude.mag if magnitude else None,
            'id': event_id
        }
        
        events_data.append(event_dict)
    
    # Create DataFrame
    df = pd.DataFrame(events_data)
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def create_event_intervals(events_df):
    """
    Create time intervals for each event (label=1) with random durations.
    Start offset: uniform(-10, -90) seconds before event
    End offset: uniform(-30, 120) seconds after event
    
    Parameters:
    -----------
    events_df : pd.DataFrame
        DataFrame with event data including 'date' column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: id, date, start_time, end_time, duration, lat, long, mag, label
    """
    intervals = []
    
    for _, event in events_df.iterrows():
        # Parse the date if it's a string
        if isinstance(event['date'], str):
            event_time = pd.to_datetime(event['date'])
        else:
            event_time = event['date']
        
        # Calculate interval boundaries with random offsets
        start_offset = np.random.uniform(10, 90)  # seconds before event
        end_offset = np.random.uniform(-30, 120)  # seconds after event
        
        start_time = event_time - timedelta(seconds=start_offset)
        end_time = event_time + timedelta(seconds=end_offset)
        
        # Calculate duration
        duration = (end_time - start_time).total_seconds()
        
        # Create interval dictionary
        interval_dict = {
            'id': event['id'],
            'date': event_time,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'lat': event['lat'],
            'long': event['long'],
            'mag': event['mag'],
            'label': 1
        }
        
        intervals.append(interval_dict)
    
    # Create DataFrame
    df = pd.DataFrame(intervals)
    
    return df


def intervals_overlap(start1, end1, start2, end2):
    """Check if two time intervals overlap."""
    return start1 <= end2 and start2 <= end1


def generate_non_event_intervals(event_intervals_df, num_intervals=None):
    """
    Generate random non-event intervals (label=0) where no events occur.
    Each interval has a random duration matching the event interval distributions.
    
    Parameters:
    -----------
    event_intervals_df : pd.DataFrame
        DataFrame with event intervals
    num_intervals : int, optional
        Number of non-event intervals to generate. If None, generates same number as events.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with non-event intervals
    """
    if num_intervals is None:
        num_intervals = len(event_intervals_df)
    
    # Convert to datetime
    global_start = pd.to_datetime(START_TIME.datetime)
    global_end = pd.to_datetime(END_TIME.datetime)
    
    # Convert event intervals to datetime
    event_starts = pd.to_datetime(event_intervals_df['start_time'])
    event_ends = pd.to_datetime(event_intervals_df['end_time'])
    
    non_event_intervals = []
    max_attempts = num_intervals * 100  # Prevent infinite loop
    attempts = 0
    
    print(f"Generating {num_intervals} non-event intervals with random durations...")
    
    while len(non_event_intervals) < num_intervals and attempts < max_attempts:
        attempts += 1
        
        # Generate random duration similar to event intervals
        # Start offset: uniform(10, 90), End offset: uniform(-30, 120)
        # This gives durations roughly in the range of (10+(-30)) to (90+120) = -20 to 210 seconds
        # But we want positive durations, so let's use a range similar to event intervals
        start_offset = np.random.uniform(10, 90)
        end_offset = np.random.uniform(-30, 120)
        duration = start_offset + end_offset  # Total duration
        
        # Ensure minimum duration of 40 seconds
        if duration < 40:
            duration = 40
        
        # Generate random start time
        time_range_seconds = (global_end - global_start).total_seconds()
        random_offset = np.random.uniform(0, time_range_seconds - duration)
        random_start = global_start + timedelta(seconds=random_offset)
        random_end = random_start + timedelta(seconds=duration)
        
        # Check if this interval overlaps with any event interval
        overlaps = False
        for i in range(len(event_intervals_df)):
            if intervals_overlap(random_start, random_end, event_starts.iloc[i], event_ends.iloc[i]):
                overlaps = True
                break
        
        # If no overlap, add to non-event intervals
        if not overlaps:
            interval_dict = {
                'id': f"non_event_{len(non_event_intervals) + 1}",
                'date': random_start + timedelta(seconds=duration/2),  # Middle of interval
                'start_time': random_start,
                'end_time': random_end,
                'duration': duration,
                'lat': None,
                'long': None,
                'mag': None,
                'label': 0
            }
            non_event_intervals.append(interval_dict)
            
            if len(non_event_intervals) % 10 == 0:
                print(f"  Generated {len(non_event_intervals)}/{num_intervals} non-event intervals")
    
    if len(non_event_intervals) < num_intervals:
        print(f"⚠ Warning: Could only generate {len(non_event_intervals)} non-overlapping intervals")
    
    df = pd.DataFrame(non_event_intervals)
    return df


def main():
    """Main execution function."""
    print("=" * 60)
    print("Signal Intervals Generation Script")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(28)

    # Define file paths
    input_file = Path(RAW_DATA_PATH) / "events.xml"
    output_file = Path(PROCESSED_DATA_PATH) / "signal_intervals.csv"
    
    # Check if input file exists
    if not input_file.exists():
        print(f"✗ Error: Input file not found: {input_file}")
        print("  Please run get_raw_data.py first to download events.")
        return
    
    print(f"Reading events from: {input_file}")
    
    try:
        # Read the event catalog from XML
        catalog = read_events(str(input_file))
        print(f"✓ Loaded catalog with {len(catalog)} events")
        
        # Extract event data to DataFrame
        print("Extracting event data...")
        events_df = extract_event_data(catalog)
        print(f"✓ Extracted {len(events_df)} events")
        
        # Create event intervals (label=1)
        print("\nGenerating event intervals (label=1)...")
        print("  Start offset: uniform(10, 90) seconds before event")
        print("  End offset: uniform(-30, 120) seconds after event")
        event_intervals_df = create_event_intervals(events_df)
        print(f"✓ Created {len(event_intervals_df)} event intervals")
        print(f"  Duration range: {event_intervals_df['duration'].min():.1f}s to {event_intervals_df['duration'].max():.1f}s")
        print(f"  Mean duration: {event_intervals_df['duration'].mean():.1f}s")
        
        # Generate non-event intervals (label=0)
        print("\nGenerating non-event intervals (label=0)...")
        non_event_intervals_df = generate_non_event_intervals(event_intervals_df)
        print(f"✓ Created {len(non_event_intervals_df)} non-event intervals")
        print(f"  Duration range: {non_event_intervals_df['duration'].min():.1f}s to {non_event_intervals_df['duration'].max():.1f}s")
        print(f"  Mean duration: {non_event_intervals_df['duration'].mean():.1f}s")
        
        # Combine both types of intervals
        print("\nCombining all intervals...")
        all_intervals_df = pd.concat([event_intervals_df, non_event_intervals_df], ignore_index=True)
        
        # Sort by start time
        all_intervals_df = all_intervals_df.sort_values('start_time').reset_index(drop=True)
        
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        all_intervals_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Signal intervals saved to: {output_file}")
        print(f"\nDataFrame info:")
        print(f"  Total rows: {len(all_intervals_df)}")
        print(f"  Event intervals (label=1): {len(all_intervals_df[all_intervals_df['label'] == 1])}")
        print(f"  Non-event intervals (label=0): {len(all_intervals_df[all_intervals_df['label'] == 0])}")
        print(f"  Columns: {list(all_intervals_df.columns)}")
        
        print(f"\nDuration statistics:")
        print(f"  Overall duration range: {all_intervals_df['duration'].min():.1f}s to {all_intervals_df['duration'].max():.1f}s")
        print(f"  Overall mean duration: {all_intervals_df['duration'].mean():.1f}s")
        
        print(f"\nFirst few intervals:")
        print(all_intervals_df.head(10))
        
        print(f"\nLabel distribution:")
        print(all_intervals_df['label'].value_counts().sort_index())
        
        print("\n" + "=" * 60)
        print("✓ Interval generation completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Error generating intervals: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

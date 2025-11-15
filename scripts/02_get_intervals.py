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
from obspy import read_events

# Add parent directory to path to allow imports from config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, START_TIME, END_TIME
from config.config import (
    EVENT_INTERVAL_START_OFFSET_MIN, EVENT_INTERVAL_START_OFFSET_MAX,
    EVENT_INTERVAL_END_OFFSET_MIN, EVENT_INTERVAL_END_OFFSET_MAX
)
from src.preprocess import extract_event_data, create_event_intervals, generate_non_event_intervals


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
        print(f"  Start offset: uniform({EVENT_INTERVAL_START_OFFSET_MIN}, {EVENT_INTERVAL_START_OFFSET_MAX}) seconds before event")
        print(f"  End offset: uniform({EVENT_INTERVAL_END_OFFSET_MIN}, {EVENT_INTERVAL_END_OFFSET_MAX}) seconds after event")
        event_intervals_df = create_event_intervals(events_df)
        print(f"✓ Created {len(event_intervals_df)} event intervals")
        print(f"  Duration range: {event_intervals_df['duration'].min():.1f}s to {event_intervals_df['duration'].max():.1f}s")
        print(f"  Mean duration: {event_intervals_df['duration'].mean():.1f}s")
        
        # Generate non-event intervals (label=0)
        print("\nGenerating non-event intervals (label=0)...")
        non_event_intervals_df = generate_non_event_intervals(
            event_intervals_df,
            global_start=START_TIME.datetime,
            global_end=END_TIME.datetime
        )
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

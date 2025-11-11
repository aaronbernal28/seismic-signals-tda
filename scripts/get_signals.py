"""
Script to fetch seismic waveforms for each interval and store in HDF5 format.
Reads intervals from data/processed/intervals.csv and saves signals to data/processed/signals.hdf5
"""

import sys
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# Add parent directory to path to allow imports from config and src
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import PROCESSED_DATA_PATH, DATA_CENTER, STATION_CODE, NETWORK, CHANNEL

def waveforms(starttime, endtime, client):
    """Download seismic waveform data."""

    print(f"Fetching waveform data for station {STATION_CODE}...")
    try:
        st = client.get_waveforms(
            network=NETWORK,
            station=STATION_CODE,
            location="*",
            channel=CHANNEL,
            starttime=starttime,
            endtime=endtime
        )
        
        print(f"✓ Waveform data fetched: {st}")
        return st

    except Exception as e:
        print(f"✗ Error downloading waveforms: {e}")
        return None

def fetch_and_store_signals(intervals_df, output_file):
    """
    Fetch waveforms for each interval and store in HDF5 file.
    
    Parameters:
    -----------
    intervals_df : pd.DataFrame
        DataFrame with interval information including start_time and end_time
    output_file : Path
        Path to output HDF5 file
    """
    # Initialize FDSN client
    print("Initializing FDSN client...")
    client = Client(DATA_CENTER)
    
    # Create HDF5 file
    with h5py.File(output_file, 'w') as hf:
        # Store metadata
        metadata_group = hf.create_group('metadata')
        
        # Convert DataFrame to store in HDF5
        for col in intervals_df.columns:
            if col in ['start_time', 'end_time', 'date']:
                # Convert datetime to string for storage
                metadata_group.create_dataset(col, data=intervals_df[col].astype(str).values.astype('S'))
            else:
                # Handle other columns
                col_data = intervals_df[col].values
                if col_data.dtype == 'object':
                    # Convert object to string
                    col_data = np.array([str(x) if x is not None else '' for x in col_data], dtype='S')
                metadata_group.create_dataset(col, data=col_data)
        
        # Create signals group
        signals_group = hf.create_group('signals')
        
        # Fetch waveforms for each interval
        total_intervals = len(intervals_df)
        successful = 0
        failed = 0
        
        print(f"\nFetching waveforms for {total_intervals} intervals...")
        print("-" * 60)
        
        for idx, row in intervals_df.iterrows():
            interval_id = row['id']
            start_time = UTCDateTime(pd.to_datetime(row['start_time']))
            end_time = UTCDateTime(pd.to_datetime(row['end_time']))
            label = row['label']
            
            print(f"[{idx+1}/{total_intervals}] Processing interval {interval_id} (label={label})...")
            
            try:
                # Fetch waveform using utils function
                st = waveforms(start_time, end_time, client)
                
                if st is not None and len(st) > 0:
                    # Process each trace in the stream
                    interval_group = signals_group.create_group(str(interval_id))
                    
                    for trace_idx, trace in enumerate(st):
                        # Store trace data
                        trace_group = interval_group.create_group(f'trace_{trace_idx}')
                        trace_group.create_dataset('data', data=trace.data, compression='gzip')
                        
                        # Store trace metadata
                        trace_group.attrs['sampling_rate'] = trace.stats.sampling_rate
                        trace_group.attrs['npts'] = trace.stats.npts
                        trace_group.attrs['network'] = trace.stats.network
                        trace_group.attrs['station'] = trace.stats.station
                        trace_group.attrs['location'] = trace.stats.location
                        trace_group.attrs['channel'] = trace.stats.channel
                        trace_group.attrs['starttime'] = str(trace.stats.starttime)
                        trace_group.attrs['endtime'] = str(trace.stats.endtime)
                    
                    # Store interval-level metadata
                    interval_group.attrs['label'] = label
                    interval_group.attrs['num_traces'] = len(st)
                    interval_group.attrs['interval_id'] = str(interval_id)
                    
                    successful += 1
                    print(f"  ✓ Stored {len(st)} trace(s)")
                else:
                    failed += 1
                    print(f"  ✗ No data retrieved")
                    
            except Exception as e:
                failed += 1
                print(f"  ✗ Error: {e}")
            
            # Progress update every 10 intervals
            if (idx + 1) % 10 == 0:
                print(f"\nProgress: {idx+1}/{total_intervals} intervals processed")
                print(f"  Success: {successful}, Failed: {failed}")
                print("-" * 60)
        
        print(f"\n{'=' * 60}")
        print(f"Summary:")
        print(f"  Total intervals: {total_intervals}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Success rate: {successful/total_intervals*100:.1f}%")
        print(f"{'=' * 60}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Signal Sampling Script")
    print("=" * 60)
    
    # Define file paths
    input_file = Path(PROCESSED_DATA_PATH) / "intervals.csv"
    output_file = Path(PROCESSED_DATA_PATH) / "signals.hdf5"
    
    # Check if input file exists
    if not input_file.exists():
        print(f"✗ Error: Input file not found: {input_file}")
        print("  Please ensure intervals.csv exists in data/processed/")
        print("  (Run get_intervals.py or get_signal_intervals.py first)")
        return
    
    print(f"Reading intervals from: {input_file}")
    
    try:
        # Read intervals CSV
        intervals_df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(intervals_df)} intervals")
        print(f"  Columns: {list(intervals_df.columns)}")
        print(f"  Labels: {intervals_df['label'].value_counts().to_dict()}")
        
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Fetch and store signals
        fetch_and_store_signals(intervals_df, output_file)
        
        print(f"\n✓ Signals saved to: {output_file}")
        print(f"  File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
        
        print("\n" + "=" * 60)
        print("✓ Signal sampling completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

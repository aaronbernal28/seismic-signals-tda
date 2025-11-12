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

from config.config import PROCESSED_DATA_PATH, DATA_CENTER, STATION_CODE, NETWORK, CHANNEL, LATITUDE, LONGITUDE

def waveforms(starttime, endtime, client):
    """Download seismic waveform data."""
    try:
        st = client.get_waveforms(
            network=NETWORK,
            station=STATION_CODE,
            location="*",
            channel=CHANNEL,
            starttime=starttime,
            endtime=endtime
        )
        return st
    except Exception as e:
        raise e


def fetch_and_store_signals(intervals_df, output_file, split_name="", batch_size=50, max_workers=3):
    """
    Fetch waveforms for each interval and store in HDF5 file.
    Uses parallel downloading with smaller batches for efficiency.
    
    Parameters:
    -----------
    intervals_df : pd.DataFrame
        DataFrame with interval information including start_time and end_time
    output_file : Path
        Path to output HDF5 file
    split_name : str
        Name of the split (e.g., "train", "test", "eval") for logging
    batch_size : int
        Number of intervals to process in parallel (default: 50)
    max_workers : int
        Maximum number of parallel download workers (default: 3)
    """
    print(f"\nProcessing {split_name.upper()} set..." if split_name else "Processing intervals...")
    
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
        
        total_intervals = len(intervals_df)
        successful = 0
        failed = 0
        
        print(f"Fetching waveforms for {total_intervals} intervals...")
        print(f"Using parallel download with {max_workers} workers, batch size: {batch_size}")
        print("-" * 60)
        
        # Process in parallel batches
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def download_single_interval(row):
            """Download a single interval's waveform."""
            interval_id = row['id']
            start_time = UTCDateTime(pd.to_datetime(row['start_time']))
            end_time = UTCDateTime(pd.to_datetime(row['end_time']))
            label = row['label']
            
            try:
                client = Client(DATA_CENTER)
                st = waveforms(start_time, end_time, client)
                return interval_id, label, st, None
            except Exception as e:
                return interval_id, label, None, str(e)
        
        # Process in batches
        for batch_start in range(0, total_intervals, batch_size):
            batch_end = min(batch_start + batch_size, total_intervals)
            batch_df = intervals_df.iloc[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start+1}-{batch_end} of {total_intervals}...")
            
            # Submit batch downloads
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_row = {
                    executor.submit(download_single_interval, row): (idx, row)
                    for idx, row in batch_df.iterrows()
                }
                
                # Process completed downloads
                for future in as_completed(future_to_row):
                    idx, row = future_to_row[future]
                    try:
                        interval_id, label, st, error = future.result()
                        
                        if st is not None and len(st) > 0:
                            # Store in HDF5
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
                            print(f"  ✓ [{successful+failed}/{total_intervals}] Interval {interval_id}: {len(st)} trace(s)")
                        else:
                            failed += 1
                            error_msg = error[:80] + "..." if error and len(error) > 80 else error
                            print(f"  ✗ [{successful+failed}/{total_intervals}] Interval {interval_id}: {error_msg or 'No data'}")
                    
                    except Exception as e:
                        failed += 1
                        print(f"  ✗ [{successful+failed}/{total_intervals}] Error processing: {str(e)[:80]}")
            
            # Progress update after each batch
            print(f"\nBatch complete. Progress: {batch_end}/{total_intervals}")
            print(f"  Success: {successful}, Failed: {failed}, Rate: {successful/(successful+failed)*100:.1f}%")
            print("-" * 60)
        
        print(f"\n{'=' * 60}")
        print(f"Summary:")
        print(f"  Total intervals: {total_intervals}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Success rate: {successful/total_intervals*100:.1f}%")
        print(f"{'=' * 60}")


def split_data(intervals_df, train_ratio=0.8, test_ratio=0.2, eval_ratio=0.0, create_eval=False):
    """
    Split data into train, test, and optionally eval sets while maintaining label balance.
    
    Parameters:
    -----------
    intervals_df : pd.DataFrame
        DataFrame with interval information
    train_ratio : float
        Proportion of data for training (default: 0.7)
    test_ratio : float
        Proportion of data for testing (default: 0.2)
    eval_ratio : float
        Proportion of data for evaluation (default: 0.1)
    create_eval : bool
        Whether to create evaluation set (default: False)
    
    Returns:
    --------
    dict: Dictionary with keys 'train', 'test', and optionally 'eval' containing DataFrames
    """
    from sklearn.model_selection import train_test_split
    
    if not create_eval:
        # Adjust ratios if no eval set
        test_ratio_adjusted = test_ratio / (train_ratio + test_ratio)
        
        # Split by label to maintain balance
        train_data = []
        test_data = []
        
        for label in intervals_df['label'].unique():
            label_data = intervals_df[intervals_df['label'] == label]
            train_label, test_label = train_test_split(
                label_data, test_size=test_ratio_adjusted, random_state=42
            )
            train_data.append(train_label)
            test_data.append(test_label)
        
        return {
            'train': pd.concat(train_data).reset_index(drop=True),
            'test': pd.concat(test_data).reset_index(drop=True)
        }
    else:
        # Three-way split
        test_eval_ratio = test_ratio + eval_ratio
        eval_ratio_adjusted = eval_ratio / test_eval_ratio
        
        train_data = []
        test_data = []
        eval_data = []
        
        for label in intervals_df['label'].unique():
            label_data = intervals_df[intervals_df['label'] == label]
            
            # First split: train vs (test + eval)
            train_label, test_eval_label = train_test_split(
                label_data, test_size=test_eval_ratio, random_state=42
            )
            
            # Second split: test vs eval
            test_label, eval_label = train_test_split(
                test_eval_label, test_size=eval_ratio_adjusted, random_state=42
            )
            
            train_data.append(train_label)
            test_data.append(test_label)
            eval_data.append(eval_label)
        
        return {
            'train': pd.concat(train_data).reset_index(drop=True),
            'test': pd.concat(test_data).reset_index(drop=True),
            'eval': pd.concat(eval_data).reset_index(drop=True)
        }


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch and split seismic signals into train/test/eval sets')
    parser.add_argument('--create-eval', action='store_true', default=False,
                        help='Create evaluation set (default: False)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='Test set ratio (default: 0.2)')
    parser.add_argument('--eval-ratio', type=float, default=0.0,
                        help='Evaluation set ratio (default: 0.0)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of intervals to process in parallel (default: 50)')
    parser.add_argument('--max-workers', type=int, default=3,
                        help='Maximum parallel download workers (default: 3)')

    args = parser.parse_args()
    
    print("=" * 60)
    print("Signal Sampling Script with Train/Test/Eval Split")
    print("=" * 60)
    
    # Define file paths
    input_file = Path(PROCESSED_DATA_PATH) / "intervals.csv"
    
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
        
        # Split data
        print(f"\nSplitting data (train={args.train_ratio}, test={args.test_ratio}" + 
              (f", eval={args.eval_ratio}" if args.create_eval else "") + ")...")
        splits = split_data(
            intervals_df, 
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            eval_ratio=args.eval_ratio,
            create_eval=args.create_eval
        )
        
        for split_name, split_df in splits.items():
            print(f"\n{split_name.upper()} set: {len(split_df)} intervals")
            print(f"  Labels: {split_df['label'].value_counts().to_dict()}")
        
        # Create output directory if it doesn't exist
        output_dir = Path(PROCESSED_DATA_PATH)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Fetch and store signals for each split
        total_size = 0
        
        for split_name, split_df in splits.items():
            output_file = output_dir / f"signals_{split_name}.hdf5"
            print(f"\n{'=' * 60}")
            print(f"Processing {split_name.upper()} set")
            print(f"{'=' * 60}")
            
            fetch_and_store_signals(
                split_df, 
                output_file, 
                split_name, 
                batch_size=args.batch_size,
                max_workers=args.max_workers
            )
            
            file_size_mb = output_file.stat().st_size / (1024*1024)
            total_size += file_size_mb
            print(f"\n✓ {split_name.upper()} signals saved to: {output_file}")
            print(f"  File size: {file_size_mb:.2f} MB")
        
        print("\n" + "=" * 60)
        print("✓ All signal sampling completed!")
        print(f"  Total size: {total_size:.2f} MB")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
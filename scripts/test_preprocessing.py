"""
Test script to verify preprocessing is working correctly.
Run this before generating the full dataset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from config.config import DATA_CENTER, NETWORK, STATION_CODE, CHANNEL
from src.preprocess import preprocess_trace, download_waveforms
import numpy as np

def test_preprocessing():
    """Test preprocessing on a single waveform."""
    
    print("=" * 70)
    print("PREPROCESSING TEST")
    print("=" * 70)
    
    # Download a small test waveform
    print("\n1. Downloading test waveform...")
    try:
        client = Client(DATA_CENTER)
        # Get a short window of data
        start_time = UTCDateTime("2024-01-01T00:00:00")
        end_time = UTCDateTime("2024-01-01T00:05:00")
        
        st = download_waveforms(client, NETWORK, STATION_CODE, CHANNEL, start_time, end_time)
        
        if len(st) == 0:
            print("✗ No data returned!")
            return False
            
        print(f"✓ Downloaded {len(st)} trace(s)")
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False
    
    # Test preprocessing on first trace
    trace = st[0]
    
    print(f"\n2. Original trace statistics:")
    print(f"   Channel: {trace.stats.channel}")
    print(f"   Sampling rate: {trace.stats.sampling_rate} Hz")
    print(f"   Number of points: {trace.stats.npts}")
    print(f"   Data range: [{np.min(trace.data):.2e}, {np.max(trace.data):.2e}]")
    print(f"   Mean: {np.mean(trace.data):.2e}")
    print(f"   Std: {np.std(trace.data):.2e}")
    
    # Check if response information is available
    print(f"\n3. Checking instrument response metadata:")
    try:
        if hasattr(trace.stats, 'response') and trace.stats.response is not None:
            print(f"   ✓ Response metadata is available")
        else:
            print(f"   ⚠ Response metadata not attached to trace")
            print(f"   Trying to fetch from client...")
            try:
                # Try to get inventory with response
                inventory = client.get_stations(
                    network=NETWORK,
                    station=STATION_CODE,
                    channel=CHANNEL,
                    starttime=start_time,
                    endtime=end_time,
                    level="response"
                )
                trace.stats.response = inventory.get_response(trace.id, start_time)
                print(f"   ✓ Response metadata fetched successfully")
            except Exception as e:
                print(f"   ✗ Could not fetch response: {e}")
                print(f"   Preprocessing will fail at remove_response step!")
    except Exception as e:
        print(f"   ✗ Error checking response: {e}")
    
    # Apply preprocessing
    print(f"\n4. Applying preprocessing...")
    preprocessed_trace, success = preprocess_trace(trace)
    
    print(f"\n5. Preprocessed trace statistics:")
    print(f"   Preprocessing success: {success}")
    print(f"   Data range: [{np.min(preprocessed_trace.data):.2e}, {np.max(preprocessed_trace.data):.2e}]")
    print(f"   Mean: {np.mean(preprocessed_trace.data):.2e}")
    print(f"   Std: {np.std(preprocessed_trace.data):.2e}")
    
    # Analyze results
    print(f"\n6. Analysis:")
    
    original_range = np.max(np.abs(trace.data))
    preprocessed_range = np.max(np.abs(preprocessed_trace.data))
    
    print(f"   Original amplitude range: {original_range:.2e}")
    print(f"   Preprocessed amplitude range: {preprocessed_range:.2e}")
    print(f"   Reduction factor: {original_range/preprocessed_range:.2e}")
    
    if success:
        if preprocessed_range < 1e-3:
            print(f"\n   ✓ SUCCESS: Data converted to physical units (displacement in meters)")
            print(f"   Values are in expected range for displacement (< 1e-3 m)")
        else:
            print(f"\n   ⚠ WARNING: Preprocessing succeeded but values still large!")
            print(f"   Expected: < 1e-3 m, Got: {preprocessed_range:.2e}")
    else:
        if preprocessed_range < original_range * 0.1:
            print(f"\n   ⚠ PARTIAL SUCCESS: Data was detrended but response not removed")
            print(f"   Values reduced but still in raw counts (not physical units)")
        else:
            print(f"\n   ✗ FAILURE: Data appears unchanged")
    
    print("\n" + "=" * 70)
    return success

if __name__ == "__main__":
    success = test_preprocessing()
    
    if success:
        print("\n✓ Preprocessing is working correctly!")
        print("  You can now run: python scripts/03_get_signals.py")
    else:
        print("\n⚠ Preprocessing has issues!")
        print("  Common causes:")
        print("  1. Instrument response metadata not available from FDSN server")
        print("  2. Network/station/channel doesn't have response information")
        print("  3. Time period requested has no response metadata")
        print("\n  Solutions:")
        print("  - Check if your data center provides response information")
        print("  - Try a different time period or station")
        print("  - Consider using data with known response metadata")
        print("\n  Note: Data will still be detrended, just not converted to physical units")

from pathlib import Path
from obspy.clients.fdsn import Client
from config.config import *


def download_waveforms():
    """Download seismic waveform data."""
    print("Initializing FDSN client...")
    client = Client(DATA_CENTER)
    
    print(f"Fetching waveform data for station {STATION_CODE}...")
    try:
        st = client.get_waveforms(
            network=NETWORK,
            station=STATION_CODE,
            location="*",
            channel=CHANNEL,
            starttime=START_TIME,
            endtime=END_TIME
        )
        
        # Save to file
        output_file = Path(RAW_DATA_PATH) / "waveforms.mseed"
        st.write(str(output_file), format='MSEED')
        
        print(f"✓ Waveform data saved to {output_file}")
        print(f"  {len(st)} traces downloaded")
        return st
        
    except Exception as e:
        print(f"✗ Error downloading waveforms: {e}")
        return None
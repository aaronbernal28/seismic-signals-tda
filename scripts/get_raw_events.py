"""
Script to download raw seismic data and event catalogs.
Based on notebook 01_experiments.ipynb
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports from config
sys.path.insert(0, str(Path(__file__).parent.parent))

from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from obspy.core.event import Catalog
from config.config import *


def download_events():
    """Download earthquake event catalog."""
    print("\nSearching for earthquake events...")
    client = Client(DATA_CENTER)
    
    window = 5  # degrees
    
    try:
        catalog = client.get_events(
            starttime=START_TIME,
            endtime=END_TIME,
            minlatitude=LATITUDE - window,
            maxlatitude=LATITUDE + window,
            minlongitude=LONGITUDE - window,
            maxlongitude=LONGITUDE + window
        )
        
        print(f"  Found {len(catalog)} events")
        
        # Filter by distance
        filtered_catalog = filter_events_by_distance(catalog)
        
        # Save to file
        output_file = Path(RAW_DATA_PATH) / "events.xml"
        filtered_catalog_obj = Catalog(events=filtered_catalog)
        filtered_catalog_obj.write(str(output_file), format='QUAKEML')
        
        print(f"✓ Event catalog saved to {output_file}")
        print(f"  {len(filtered_catalog)} events after distance filtering")
        return filtered_catalog_obj
        
    except Exception as e:
        print(f"✗ Error downloading events: {e}")
        return None


def filter_events_by_distance(catalog):
    """Filter events by distance from station."""
    max_distance_degrees = MAX_DISTANCE_KM / 111.19
    filtered = []
    
    for event in catalog:
        origin = event.preferred_origin() or event.origins[0]
        eq_lat = origin.latitude
        eq_lon = origin.longitude
        
        distance_degrees = locations2degrees(eq_lat, eq_lon, LATITUDE, LONGITUDE)
        
        if distance_degrees <= max_distance_degrees:
            filtered.append(event)
    
    return filtered


def main():
    """Main execution function."""
    print("=" * 60)
    print("Seismic Data Acquisition Script")
    print("=" * 60)
    print(f"Station: {NETWORK}.{STATION_CODE}")
    print(f"Time range: {START_TIME} to {END_TIME}")
    print(f"Channel: {CHANNEL}")
    print("=" * 60)
    
    # Download data
    events = download_events()
    
    print("\n" + "=" * 60)
    if events:
        print("✓ Data acquisition completed successfully!")
    else:
        print("⚠ Data acquisition completed with errors")
    print("=" * 60)


if __name__ == "__main__":
    main()

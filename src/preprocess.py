"""
Reusable preprocessing and data handling functions for seismic signal processing.
These functions are used across multiple scripts for event handling, interval generation,
and signal downloading/preprocessing.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from obspy.core.event import Catalog
from obspy import UTCDateTime
import warnings
from config.config import (
    EVENT_INTERVAL_START_OFFSET_MIN,
    EVENT_INTERVAL_START_OFFSET_MAX,
    EVENT_INTERVAL_END_OFFSET_MIN,
    EVENT_INTERVAL_END_OFFSET_MAX,
    NON_EVENT_INTERVAL_DURATION
)


# ============================================================================
# EVENT HANDLING FUNCTIONS (from 01_get_raw_events.py)
# ============================================================================

def download_events(client, start_time, end_time, latitude, longitude, window_degrees=5):
    """
    Download earthquake event catalog from FDSN client.
    
    Parameters:
    -----------
    client : obspy.clients.fdsn.Client
        FDSN client instance
    start_time : obspy.UTCDateTime
        Start time for event search
    end_time : obspy.UTCDateTime
        End time for event search
    latitude : float
        Station latitude
    longitude : float
        Station longitude
    window_degrees : float
        Search window in degrees (default: 5)
        
    Returns:
    --------
    obspy.core.event.Catalog : Downloaded event catalog
    
    Raises:
    -------
    Exception : If download fails
    """
    try:
        catalog = client.get_events(
            starttime=start_time,
            endtime=end_time,
            minlatitude=latitude - window_degrees,
            maxlatitude=latitude + window_degrees,
            minlongitude=longitude - window_degrees,
            maxlongitude=longitude + window_degrees
        )
        return catalog
    except Exception as e:
        raise e


def filter_events_by_distance(catalog, station_lat, station_lon, max_distance_km):
    """
    Filter events by distance from station.
    
    Parameters:
    -----------
    catalog : obspy.core.event.Catalog
        Event catalog to filter
    station_lat : float
        Station latitude
    station_lon : float
        Station longitude
    max_distance_km : float
        Maximum distance in kilometers
        
    Returns:
    --------
    list : List of filtered events
    """
    max_distance_degrees = max_distance_km / 111.19
    filtered = []
    
    for event in catalog:
        origin = event.preferred_origin() or event.origins[0]
        eq_lat = origin.latitude
        eq_lon = origin.longitude
        
        distance_degrees = locations2degrees(eq_lat, eq_lon, station_lat, station_lon)
        
        if distance_degrees <= max_distance_degrees:
            filtered.append(event)
    
    return filtered


# ============================================================================
# INTERVAL GENERATION FUNCTIONS (from 02_get_intervals.py)
# ============================================================================

def extract_event_data(catalog):
    """
    Extract event information from ObsPy catalog.
    
    Parameters:
    -----------
    catalog : obspy.core.event.Catalog
        The event catalog to process
        
    Returns:
    --------
    pd.DataFrame : DataFrame with columns: date, lat, long, mag, id
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
    Start offset: uniform(EVENT_INTERVAL_START_OFFSET_MIN, EVENT_INTERVAL_START_OFFSET_MAX) seconds before event
    End offset: uniform(EVENT_INTERVAL_END_OFFSET_MIN, EVENT_INTERVAL_END_OFFSET_MAX) seconds after event
    
    Parameters:
    -----------
    events_df : pd.DataFrame
        DataFrame with event data including 'date' column
        
    Returns:
    --------
    pd.DataFrame : DataFrame with columns: id, date, start_time, end_time, duration, lat, long, mag, label
    """
    intervals = []
    
    for _, event in events_df.iterrows():
        # Parse the date if it's a string
        if isinstance(event['date'], str):
            event_time = pd.to_datetime(event['date'])
        else:
            event_time = event['date']
        
        # Calculate interval boundaries with random offsets
        start_offset = np.random.uniform(EVENT_INTERVAL_START_OFFSET_MIN, EVENT_INTERVAL_START_OFFSET_MAX)  # seconds before event
        end_offset = np.random.uniform(EVENT_INTERVAL_END_OFFSET_MIN, EVENT_INTERVAL_END_OFFSET_MAX)  # seconds after event
        
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
    """
    Check if two time intervals overlap.
    
    Parameters:
    -----------
    start1, end1 : datetime
        First interval boundaries
    start2, end2 : datetime
        Second interval boundaries
        
    Returns:
    --------
    bool : True if intervals overlap, False otherwise
    """
    return start1 <= end2 and start2 <= end1


def generate_non_event_intervals(event_intervals_df, global_start, global_end, num_intervals=None):
    """
    Generate random non-event intervals (label=0) where no events occur.
    Each interval has a random duration matching the event interval distributions.
    
    Parameters:
    -----------
    event_intervals_df : pd.DataFrame
        DataFrame with event intervals
    global_start : datetime
        Start of the global time range
    global_end : datetime
        End of the global time range
    num_intervals : int, optional
        Number of non-event intervals to generate. If None, generates same number as events.
        
    Returns:
    --------
    pd.DataFrame : DataFrame with non-event intervals
    """
    if num_intervals is None:
        num_intervals = len(event_intervals_df)
    
    # Convert to datetime
    global_start = pd.to_datetime(global_start)
    global_end = pd.to_datetime(global_end)
    
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
        start_offset = np.random.uniform(EVENT_INTERVAL_START_OFFSET_MIN, EVENT_INTERVAL_START_OFFSET_MAX)
        end_offset = np.random.uniform(EVENT_INTERVAL_END_OFFSET_MIN, EVENT_INTERVAL_END_OFFSET_MAX)
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


# ============================================================================
# SIGNAL PROCESSING FUNCTIONS (from 03_get_signals.py)
# ============================================================================

def preprocess_trace(trace):
    """
    Apply preprocessing steps to a seismic trace.
    
    Steps:
    1. Detrend - remove mean
    2. Detrend - remove linear trend
    3. Remove instrument response (convert to displacement)
    
    Parameters:
    -----------
    trace : obspy.Trace
        Input seismic trace
        
    Returns:
    --------
    tuple: (preprocessed_trace, success_flag)
        - preprocessed_trace: obspy.Trace - Preprocessed trace
        - success_flag: bool - True if all steps succeeded, False otherwise
    """
    # Make a copy to avoid modifying original
    trace_copy = trace.copy()
    preprocessing_success = True
    
    try:
        # Step 1: Remove mean
        trace_copy.detrend('demean')
        
        # Step 2: Remove linear trend
        trace_copy.detrend('linear')
        
    except Exception as e:
        print(f"    ⚠ Warning: Detrending failed for trace {trace.id}: {str(e)[:80]}")
        preprocessing_success = False
        return trace, False  # Return original if basic detrending fails
    
    # Step 3: Remove instrument response (convert to displacement)
    # This step is most likely to fail if metadata is missing
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace_copy.remove_response(output='DISP')
    except Exception as e:
        print(f"    ⚠ Warning: remove_response failed for trace {trace.id}: {str(e)[:80]}")
        print(f"       Continuing with detrended data only (not converted to physical units)")
        preprocessing_success = False
        # Return the detrended trace even if response removal fails
    
    return trace_copy, preprocessing_success


def download_waveforms(client, network, station, channel, start_time, end_time):
    """
    Download seismic waveform data from FDSN client.
    
    Parameters:
    -----------
    client : obspy.clients.fdsn.Client
        FDSN client instance
    network : str
        Network code
    station : str
        Station code
    channel : str
        Channel code
    start_time : obspy.UTCDateTime
        Start time
    end_time : obspy.UTCDateTime
        End time
        
    Returns:
    --------
    obspy.Stream : Downloaded waveform stream
    
    Raises:
    -------
    Exception : If download fails
    """
    try:
        st = client.get_waveforms(
            network=network,
            station=station,
            location="*",
            channel=channel,
            starttime=start_time,
            endtime=end_time
        )
        return st
    except Exception as e:
        raise e


def split_data(intervals_df, train_ratio=0.8, test_ratio=0.2, eval_ratio=0.0, create_eval=False, random_state=42):
    """
    Split data into train, test, and optionally eval sets while maintaining label balance.
    
    Parameters:
    -----------
    intervals_df : pd.DataFrame
        DataFrame with interval information
    train_ratio : float
        Proportion of data for training (default: 0.8)
    test_ratio : float
        Proportion of data for testing (default: 0.2)
    eval_ratio : float
        Proportion of data for evaluation (default: 0.0)
    create_eval : bool
        Whether to create evaluation set (default: False)
    random_state : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict : Dictionary with keys 'train', 'test', and optionally 'eval' containing DataFrames
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
                label_data, test_size=test_ratio_adjusted, random_state=random_state
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
                label_data, test_size=test_eval_ratio, random_state=random_state
            )
            
            # Second split: test vs eval
            test_label, eval_label = train_test_split(
                test_eval_label, test_size=eval_ratio_adjusted, random_state=random_state
            )
            
            train_data.append(train_label)
            test_data.append(test_label)
            eval_data.append(eval_label)
        
        return {
            'train': pd.concat(train_data).reset_index(drop=True),
            'test': pd.concat(test_data).reset_index(drop=True),
            'eval': pd.concat(eval_data).reset_index(drop=True)
        }

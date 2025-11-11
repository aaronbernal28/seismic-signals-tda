from obspy import UTCDateTime

# Station parameters
DATA_CENTER = "IRIS"
STATION_CODE = "GO01"
NETWORK = "C"
LATITUDE = -19.6685
LONGITUDE = -69.1942

# Time parameters from Jan 1, 2025 to July 1, 2025
START_TIME = UTCDateTime("2025-01-01T00:00:00")
END_TIME = UTCDateTime("2025-07-01T00:00:00")

# Analysis parameters
CHANNEL = "BHZ"
MIN_MAGNITUDE = 4.0
MAX_DISTANCE_KM = 150

# Data storage paths
RAW_DATA_PATH = "./data/raw/"
PROCESSED_DATA_PATH = "./data/processed/"
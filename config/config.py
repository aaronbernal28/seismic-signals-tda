from obspy import UTCDateTime

# Station parameters
DATA_CENTER = "IRIS"
STATION_CODE = "GO01"
NETWORK = "C"
LATITUDE = -19.6685
LONGITUDE = -69.1942
SR = 40 # Sampling rate

# Time parameters from Nov 1, 2023 to November 1, 2025
START_TIME = UTCDateTime("2024-11-01T00:00:00")#UTCDateTime("2023-11-01T00:00:00")
END_TIME = UTCDateTime("2025-11-01T00:00:00")

# Analysis parameters
CHANNEL = "BHZ"
MIN_MAGNITUDE = None  # No minimum magnitude filter
MAX_DISTANCE_KM = 150

# Data storage paths
RAW_DATA_PATH = "./data/raw/"
PROCESSED_DATA_PATH = "./data/processed/"

# Train and test paths
TRAIN_DATA_PATH = "./data/processed/signals_train.hdf5"
TEST_DATA_PATH = "./data/processed/signals_test.hdf5"
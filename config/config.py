import ast
import csv
import sys
from pathlib import Path
from obspy import UTCDateTime
from persim import wasserstein
import math

# Station parameters
DATA_CENTER = "IRIS"
STATION_CODE = "GO01"
NETWORK = "C"
LATITUDE = -19.6685
LONGITUDE = -69.1942
SR = 40 # Sampling rate

# Time parameters from Nov 1, 2023 to November 1, 2025
START_TIME = UTCDateTime("2023-11-01T00:00:00")#UTCDateTime("2023-11-01T00:00:00")
END_TIME = UTCDateTime("2025-11-01T00:00:00")

# Analysis parameters
CHANNEL = "BHZ"
MIN_MAGNITUDE = None  # No minimum magnitude filter
MAX_DISTANCE_KM = 300  # Maximum distance from station in km

# Data storage paths
RAW_DATA_PATH = "./data/raw/"
PROCESSED_DATA_PATH = "./data/processed/"

# Train and test paths
TRAIN_DATA_PATH = "./data/processed/signals_train.hdf5"
TEST_DATA_PATH = "./data/processed/signals_test.hdf5"
MAX_SAMPLES = None  # Limitar muestras cargadas (None usa todo el dataset)

# Resultados y grid search
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results" / "best_model_analysis"
GRID_SEARCH_CSV = Path(__file__).parent.parent / "data" / "results" / "grid_search_te.csv"

# Event interval generation parameters
EVENT_INTERVAL_START_OFFSET_MIN = 10    # seconds before event
EVENT_INTERVAL_START_OFFSET_MAX = 20    # seconds before event
EVENT_INTERVAL_END_OFFSET_MIN = 30     # seconds after event
EVENT_INTERVAL_END_OFFSET_MAX = 60     # seconds after event
NON_EVENT_INTERVAL_DURATION = 60 # seconds

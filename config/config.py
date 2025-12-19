import ast
import csv
import sys
from pathlib import Path
from obspy import UTCDateTime
from persim import wasserstein

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
MAX_SAMPLES = 400  # Limitar muestras cargadas (None usa todo el dataset)

# Resultados y grid search
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results" / "best_model_analysis"
GRID_SEARCH_CSV = Path(__file__).parent.parent / "data" / "results" / "grid_search_te.csv"

# Event interval generation parameters
EVENT_INTERVAL_START_OFFSET_MIN = 10    # seconds before event
EVENT_INTERVAL_START_OFFSET_MAX = 20    # seconds before event
EVENT_INTERVAL_END_OFFSET_MIN = 30     # seconds after event
EVENT_INTERVAL_END_OFFSET_MAX = 60     # seconds after event
NON_EVENT_INTERVAL_DURATION = 60 # seconds


def load_best_te_params(csv_path=GRID_SEARCH_CSV):
	"""Cargar parámetros y seed del mejor modelo TE según grid_search_te.csv."""
	if not csv_path.exists():
		raise FileNotFoundError(f"No se encontró el archivo de grid search en {csv_path}")

	with csv_path.open(newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		rows = list(reader)

	if not rows:
		raise ValueError("El archivo de grid search está vacío")

	best_row = sorted(
		rows,
		key=lambda r: (
			int(float(r['rank'])),
			-float(r['mean_auc']),
			float(r['fit_time'])
		)
	)[0]

	weights = ast.literal_eval(best_row['weights'])

	params = {
		'distance': wasserstein,
		'weights': weights,
		'thresh': float(best_row['thresh']),
		'tau': int(float(best_row['tau'])),
		'stride': int(float(best_row['stride'])),
		'sample': int(float(best_row['sample'])),
		'max_points': int(float(best_row['max_points'])),
		'dim': int(float(best_row['dim'])),
		'alpha': float(best_row['alpha']),
	}

	model_seed = int(float(best_row['seed']))

	return params, model_seed


BEST_PARAMS, MODEL_SEED = load_best_te_params()
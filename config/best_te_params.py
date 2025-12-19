"""
Parámetros del mejor modelo TE cargados desde data/results/grid_search_te.csv.
Separado de config.py para evitar fallas de import cuando el CSV está incompleto.
"""
import ast
import csv
import math
from pathlib import Path
from persim import wasserstein

# Ruta al CSV de grid search
GRID_SEARCH_CSV = Path(__file__).parent.parent / "data" / "results" / "grid_search_te.csv"


def _to_float(val):
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except Exception:
        return None


def load_best_te_params(csv_path=GRID_SEARCH_CSV):
    """Cargar parámetros y seed del mejor modelo TE según grid_search_te.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de grid search en {csv_path}")

    with csv_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    if not rows:
        raise ValueError("El archivo de grid search está vacío")

    valid_rows = [
        r for r in rows
        if _to_float(r.get("mean_auc")) is not None
        and _to_float(r.get("rank")) is not None
        and _to_float(r.get("fit_time")) is not None
    ]

    if not valid_rows:
        raise ValueError("grid_search_te.csv no contiene filas con mean_auc válido")

    best_row = sorted(
        valid_rows,
        key=lambda r: (
            int(_to_float(r["rank"])),
            -_to_float(r["mean_auc"]),
            _to_float(r["fit_time"]),
        ),
    )[0]

    weights = ast.literal_eval(best_row["weights"])

    params = {
        "distance": wasserstein,
        "weights": weights,
        "thresh": float(best_row["thresh"]),
        "tau": int(float(best_row["tau"])),
        "stride": int(float(best_row["stride"])),
        "sample": int(float(best_row["sample"])),
        "max_points": int(float(best_row["max_points"])),
        "dim": int(float(best_row["dim"])),
        "alpha": float(best_row["alpha"]),
    }

    model_seed = int(float(best_row["seed"]))

    return params, model_seed


BEST_PARAMS, MODEL_SEED = load_best_te_params()

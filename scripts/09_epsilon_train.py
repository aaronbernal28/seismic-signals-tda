"""
Script: 09_epsilon_train.py
Descripcion: Calcula y grafica la distribucion del diametro de las nubes de puntos
             provenientes de las incrustaciones de Takens (conjunto de train).
             Usa los mejores parametros del modelo TE, sin recorte de puntos,
             alpha=1 y thresh=np.inf.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist

# Agregar directorio padre al path para importar modulos de src
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.utils as ut

# Conjuntos de parametros a comparar (variando dim y tau)
PARAM_GRID = [
    {"name": "d4_t4", "dim": 4, "tau": 4},
    {"name": "d5_t4", "dim": 5, "tau": 4},
    {"name": "d4_t6", "dim": 4, "tau": 6},
    {"name": "d5_t6", "dim": 5, "tau": 6},
]

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results" / "best_model_analysis"
PLOT_PATH = RESULTS_DIR / "epsilon_train_grid.png"


def compute_diameter(point_cloud: np.ndarray) -> float:
    """Diametro de la nube: maxima distancia euclidiana par-a-par.
    Si hay 0/1 puntos, retorna 0.
    """
    if point_cloud is None or len(point_cloud) <= 1:
        return 0.0
    dists = pdist(point_cloud, metric="euclidean")
    if len(dists) == 0:
        return 0.0
    return float(np.max(dists))



def process_param_set(X_train, dim, tau):
    diameters = []
    for idx, sig in enumerate(X_train):
        try:
            sig_norm = ut.normalize_minmax(sig)
            embedding = ut.takens_embedding(sig_norm, dim=dim, tau=tau)
            diam = compute_diameter(embedding)
            diameters.append(diam)
        except Exception as exc:
            print(f"ADVERTENCIA: No se pudo procesar la señal {idx} (dim={dim}, tau={tau}): {exc}")
            continue
    return np.array(diameters, dtype=float)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DISTRIBUCION DE DIAMETROS (GRID dim/tau)")
    print("=" * 70)

    # Cargar datasets (solo train) una sola vez
    X_train, y_train, _, _ = ut.load_datasets(max_samples=None)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, cfg in zip(axes, PARAM_GRID):
        # Filtrar señales demasiado cortas para Takens según cada config
        X_filt, y_filt, removed = ut.filter_valid_signals(
            X_train, y_train, dim=cfg["dim"], tau=cfg["tau"], verbose=False
        )
        print(f"Config {cfg['name']}: señales usadas {len(X_filt)}, removidas {removed}")

        diameters = process_param_set(X_filt, dim=cfg["dim"], tau=cfg["tau"])
        if len(diameters) == 0:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
            ax.set_title(f"{cfg['name']} (sin datos)")
            continue

        mean_d = float(np.mean(diameters))
        sns.histplot(diameters, bins=30, color="#1f77b4", edgecolor="white", alpha=0.8, ax=ax)

        # Lineas verticales en mean, mean/3, mean/6, mean/9 con estilo distinto
        for frac, style, w in zip([1, 1/3, 1/6, 1/9], ["-", "--", "-.", ":"], [1.6, 1.2, 1.0, 0.8]):
            val = mean_d * frac
            ax.axvline(val, color="red", linestyle=style, linewidth=w, alpha=0.9)

        ax.set_title(f"{cfg['name']} | μ={mean_d:.2f}")
        ax.set_xlabel("Diametro (max dist euclidiana)")
        ax.set_ylabel("Frecuencia")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()
    print(f"✓ Grid de histogramas guardado en {PLOT_PATH}")


if __name__ == "__main__":
    main()

"""
Script: 08_distance.py
Descripcion: Calcula la matriz de distancias bottleneck entre los diagramas de persistencia
             generados con el mejor modelo TE sobre el conjunto de prueba y grafica
             un grafo con layout de resortes coloreado por clase.
"""

import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Agregar directorio padre al path para importar modulos de src
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import RESULTS_DIR, BEST_PARAMS, MODEL_SEED, MAX_SAMPLES
from src.models.model2 import BinaryClassificationTE
import src.utils as ut

GRAPH_PATH = RESULTS_DIR / "08_distance_graph.png"


def cargar_datos_prueba():
    """Cargar conjunto de prueba y filtrar senales muy cortas."""
    _, _, X_test, y_test = ut.load_datasets(max_samples=MAX_SAMPLES)
    X_test, y_test, _ = ut.filter_valid_signals(
        X_test,
        y_test,
        dim=BEST_PARAMS["dim"],
        tau=BEST_PARAMS["tau"],
        verbose=True,
    )
    print(f"✓ Senales de prueba tras filtrado: {len(X_test)}")
    return X_test, y_test


def generar_diagramas(model, signals):
    """Generar diagramas de persistencia para cada senal de entrada."""
    model._ensure_initialized()  # Necesario para configurar TE y pesos normalizados
    diagrams = []
    valid_indices = []
    errores = 0

    for idx, sig in enumerate(signals):
        try:
            dgms = model.transform(sig)
            diagrams.append(dgms)
            valid_indices.append(idx)
        except Exception as exc:  # pragma: no cover - seguimiento basico
            errores += 1
            print(f"ADVERTENCIA: No se pudo procesar la senal {idx}: {exc}")

    if errores > 0:
        print(f"ADVERTENCIA: {errores} senal(es) se omitieron al generar diagramas.")

    return diagrams, valid_indices


def distancia_ponderada(model, dgm_a, dgm_b):
    """Calcular distancia total ponderada usando los pesos del mejor modelo."""
    total = 0.0
    for w, da, db in zip(model._normalized_weights, dgm_a, dgm_b):
        total += w * ut.bottleneck_distance(da, db)
    return total


def construir_matriz_distancias(model, diagrams):
    """Armar la matriz simetrica de distancias bottleneck entre diagramas."""
    n = len(diagrams)
    dist_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            dist = distancia_ponderada(model, diagrams[i], diagrams[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix


def construir_grafo(dist_matrix, labels, k_vecinos=50):
    """Crear grafo k-NN usando distancias como pesos."""
    n = len(labels)
    G = nx.Graph()
    for idx, label in enumerate(labels):
        G.add_node(idx, label=label)

    for i in range(n):
        vecinos_ordenados = np.argsort(dist_matrix[i])
        vecinos_validos = [j for j in vecinos_ordenados if j != i][:k_vecinos]
        for j in vecinos_validos:
            if G.has_edge(i, j):
                continue
            peso = 1.0 / (1.0 + dist_matrix[i, j])  # pesos mayores para distancias mas cortas
            G.add_edge(i, j, weight=peso, dist=dist_matrix[i, j])
    return G


def graficar_grafo(G, labels, save_path):
    """Graficar el grafo con spring layout coloreando por clase."""
    if len(G.nodes) == 0:
        print("No hay nodos para graficar.")
        return

    pos = nx.spring_layout(G, weight="weight", seed=MODEL_SEED, iterations=500)
    colores = ["#1f77b4" if labels[node] == 0 else "#d62728" for node in G.nodes]
    pesos_aristas = [1.5 * datos["weight"] for _, _, datos in G.edges(data=True)]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.25, edge_color="#999", width=pesos_aristas)
    nx.draw_networkx_nodes(G, pos, node_color=colores, node_size=50, alpha=0.7)
    plt.axis("off")
    plt.title("Grafo de distancias bottleneck (layout de resortes)", fontsize=12)

    # Leyenda manual en espanol
    from matplotlib.lines import Line2D

    leyenda = [
        Line2D([0], [0], marker="o", color="w", label="Ruido (clase 0)",
               markerfacecolor="#1f77b4", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Sismo (clase 1)",
               markerfacecolor="#d62728", markersize=8),
    ]
    plt.legend(handles=leyenda, loc="lower left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Grafo guardado en {save_path}")


def main():
    inicio = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MATRIZ DE DISTANCIAS BOTTLENECK (CONJUNTO DE PRUEBA)")
    print("=" * 70)

    X_test, y_test = cargar_datos_prueba()
    model = BinaryClassificationTE(**BEST_PARAMS, seed=MODEL_SEED)

    print("Generando diagramas de persistencia...")
    diagrams, indices_validos = generar_diagramas(model, X_test)

    # Ajustar etiquetas a las senales que se procesaron correctamente
    y_test_filtrado = y_test[indices_validos]

    print("Calculando matriz de distancias bottleneck...")
    dist_matrix = construir_matriz_distancias(model, diagrams)

    print("Construyendo y graficando grafo k-NN...")
    grafo = construir_grafo(dist_matrix, y_test_filtrado)
    graficar_grafo(grafo, y_test_filtrado, GRAPH_PATH)

    print(f"Tiempo total: {time.time() - inicio:.2f} segundos")
    print(f"Resultados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

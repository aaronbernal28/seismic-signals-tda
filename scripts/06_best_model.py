"""
Script: 06_best_model.py
Descripción: Evaluación y exploración exhaustiva del mejor modelo TDA en datos de prueba.
             Los parámetros se toman del Rango 1 de data/results/grid_search_te.csv
"""

import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix, 
    roc_curve
)
from persim import wasserstein, plot_diagrams

# Agregar directorio padre al path para importar módulos src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model2 import BinaryClassificationTE
import src.utils as ut

# Configuración para el modelo TE basada en run_grid_search.py
BEST_PARAMS = {
    'distance': ut.bottleneck_distance,
    'weights': (2, 1),
    'thresh': 1888.8889,
    'tau': 4,
    'stride': 1,
    'sample': 16,
    'max_points': 100,
    'dim': 4,
    'alpha': 0.6833
}

# La búsqueda de cuadrícula usó seed=28, así que debemos usarla aquí para reproducibilidad
MODEL_SEED = 28

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results" / "best_model_analysis"

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Generar y guardar un mapa de calor de matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Ruido', 'Terremoto'],
                yticklabels=['Ruido', 'Terremoto'])
    plt.title('Matriz de Confusión', fontsize=14)
    plt.ylabel('Etiqueta Verdadera', fontsize=12)
    plt.xlabel('Etiqueta Predicha', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Matriz de confusión guardada en {save_path}")

def plot_roc_curve(y_true, y_proba, auc_score, save_path):
    """Generar y guardar la curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Característica Operativa del Receptor (ROC)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Curva ROC guardada en {save_path}")


def plot_examples_grid(X, y, y_pred, y_proba, samples_per_class, model, model_params, save_path, seed=None, min_h0_points=10):
    """Graficar diagramas de persistencia por clase en una grilla 2xN.
    Selecciona señales con H0 suficiente y muestrea de forma estratificada por probabilidad: baja/medio/alta P(Terr).
    """
    classes = [0, 1]
    fig, axes = plt.subplots(2, samples_per_class, figsize=(3.5 * samples_per_class, 7), sharex=False)

    # Si samples_per_class == 1, axes no es 2D; normalizar
    if samples_per_class == 1:
        axes = np.array([axes]).reshape(2, 1)

    # Use seeded random sampling for reproducibility
    rng = np.random.default_rng(seed)

    for row, cls in enumerate(classes):
        idx_cls = np.where(y == cls)[0]
        
        # Filtrar índices con diagramas H0 válidos
        valid_idx_cls = []
        for idx in idx_cls:
            sig = X[idx]
            try:
                dgms = model.transform(sig)
                dgm_h0 = dgms[0]
                dgm_h0_finite = dgm_h0[dgm_h0[:, 1] != np.inf] if len(dgm_h0) > 0 else dgm_h0
                if len(dgm_h0_finite) >= min_h0_points:
                    valid_idx_cls.append(idx)
            except:
                continue
        
        # Estratificar por probabilidad P(terr) en bins: bajo, medio, alto
        chosen = []
        if len(valid_idx_cls) > 0:
            probs_cls1 = y_proba[valid_idx_cls]
            low_bin = [idx for idx, p in zip(valid_idx_cls, probs_cls1) if p <= 0.33]
            mid_bin = [idx for idx, p in zip(valid_idx_cls, probs_cls1) if 0.33 < p <= 0.67]
            high_bin = [idx for idx, p in zip(valid_idx_cls, probs_cls1) if p > 0.67]

            # Tomar uno por bin si existe
            for bin_idxs in (low_bin, mid_bin, high_bin):
                if bin_idxs:
                    chosen.append(rng.choice(bin_idxs))

            # Rellenar restantes (si faltan) de los válidos restantes
            remaining_slots = samples_per_class - len(chosen)
            if remaining_slots > 0:
                remaining_pool = [idx for idx in valid_idx_cls if idx not in chosen]
                if len(remaining_pool) > 0:
                    if len(remaining_pool) > remaining_slots:
                        extra = rng.choice(remaining_pool, size=remaining_slots, replace=False)
                        chosen.extend(list(extra))
                    else:
                        chosen.extend(remaining_pool)

        for col in range(samples_per_class):
            ax = axes[row, col]
            if col < len(chosen):
                idx = chosen[col]
                sig = X[idx]
                # Compute persistence diagrams using model's transform
                dgms = model.transform(sig)
                # Plot persistence diagrams
                plot_diagrams(dgms, ax=ax, legend=False)
                
                # Get prediction info
                true_label = 'Ruido' if cls == 0 else 'Terremoto'
                prob_noise = 1 - y_proba[idx]  # P(class 0)
                prob_earthquake = y_proba[idx]  # P(class 1)
                
                # Create clearer title
                title = f"#{idx}: {true_label}\n"
                title += f"P(Ruido)={prob_noise:.2f} | P(Terr)={prob_earthquake:.2f}"
                ax.set_title(title, fontsize=8.5)
            else:
                ax.axis('off')

            if col == 0:
                label_name = 'Ruido (Clase 0)' if cls == 0 else 'Terremoto (Clase 1)'
                ax.set_ylabel(label_name, fontsize=10, fontweight='bold')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✓ Ejemplos guardados en {save_path}")

def analyze_errors(X_test, y_test, y_pred, y_proba):
    """Realizar análisis de muestras mal clasificadas."""
    # Identificar índices
    false_positives = np.where((y_test == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_test == 1) & (y_pred == 0))[0]
    
    print(f"\n{'=' * 40}")
    print("ANÁLISIS DE ERRORES")
    print(f"{'=' * 40}")
    print(f"Falsos Positivos (Ruido predicho como Terremoto): {len(false_positives)}")
    print(f"Falsos Negativos (Terremoto predicho como Ruido): {len(false_negatives)}")
    
    if len(false_positives) > 0:
        print("\nPrincipales Falsos Positivos (predicciones incorrectas de mayor confianza):")
        # Ordenar por probabilidad de ser positivo
        fp_probs = y_proba[false_positives]
        sorted_indices = np.argsort(fp_probs)[::-1][:3] # Top 3
        
        for idx in sorted_indices:
            orig_idx = false_positives[idx]
            prob = fp_probs[idx]
            sig = X_test[orig_idx]
            print(f"  - Índice {orig_idx}: Prob(Terremoto)={prob:.4f}, Media de Señal={np.mean(sig):.2e}, Desv={np.std(sig):.2e}")

    if len(false_negatives) > 0:
        print("\nPrincipales Falsos Negativos (predicciones incorrectas de menor confianza):")
        # Ordenar por probabilidad de ser positivo (queremos baja prob aquí, es decir, confiado de que es negativo)
        fn_probs = y_proba[false_negatives]
        sorted_indices = np.argsort(fn_probs)[:3] # Bottom 3
        
        for idx in sorted_indices:
            orig_idx = false_negatives[idx]
            prob = fn_probs[idx]
            sig = X_test[orig_idx]
            print(f"  - Índice {orig_idx}: Prob(Terremoto)={prob:.4f}, Media de Señal={np.mean(sig):.2e}, Desv={np.std(sig):.2e}")

def filter_empty_diagrams(X, y, model_params, min_h1_points=15, verbose=True):
    """Filtrar señales que producen diagramas H1 vacíos o con muy pocos puntos.
    
    Args:
        X: Lista de señales
        y: Array de etiquetas
        model_params: Diccionario con parámetros del modelo
        min_h1_points: Número mínimo de puntos requeridos en H1 (por defecto 3)
        verbose: Mostrar información de filtrado
    
    Returns:
        tuple: (X_filtered, y_filtered, num_removed)
    """
    valid_indices = []
    
    for idx, signal in enumerate(X):
        try:
            # Computar embedding de Takens
            embedding = ut.takens_embedding(
                signal,
                dim=model_params['dim'],
                tau=model_params['tau']
            )
            
            # Limitar puntos si es necesario
            if len(embedding) > model_params['max_points']:
                step = len(embedding) // model_params['max_points']
                embedding = embedding[::step][:model_params['max_points']]
            
            # Computar diagramas de persistencia
            dgms = ut.compute_persistence(
                embedding,
                maxdim=1,
                thresh=model_params['thresh']
            )
            
            # Verificar si H1 tiene suficientes puntos
            dgm_h1 = dgms[1]
            dgm_h1_finite = dgm_h1[dgm_h1[:, 1] != np.inf] if len(dgm_h1) > 0 else dgm_h1
            
            # Solo mantener señales con al menos min_h1_points puntos en H1
            if len(dgm_h1_finite) >= min_h1_points:
                valid_indices.append(idx)
        except Exception as e:
            # Si hay error, excluir la señal
            if verbose:
                print(f"Error procesando señal {idx}: {e}")
            continue
    
    num_removed = len(X) - len(valid_indices)
    
    if num_removed > 0 and verbose:
        removed_by_label = {}
        for i, (signal, label) in enumerate(zip(X, y)):
            if i not in valid_indices:
                removed_by_label[label] = removed_by_label.get(label, 0) + 1
        
        print(f"\n⚠ ADVERTENCIA: {num_removed} señal(es) removida(s) (H1 con menos de {min_h1_points} puntos)")
        for label, count in removed_by_label.items():
            label_name = 'Terremoto' if label == 1 else 'Ruido'
            print(f"  - {label_name} (etiqueta {label}): {count} señal(es)")
    
    X_filtered = [X[i] for i in valid_indices]
    y_filtered = y[valid_indices]
    
    return X_filtered, y_filtered, num_removed

def main():
    """Flujo de ejecución principal."""
    # Configuración
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(MODEL_SEED)
    
    print("=" * 70)
    print("EVALUACIÓN Y EXPLORACIÓN DEL MEJOR MODELO")
    print("=" * 70)
    print("Usando parámetros del Rango 1 de grid_search_te.csv")
    print(f"Parámetros: {BEST_PARAMS}")
    
    # 1. Cargar Datos
    X_train, y_train, X_test, y_test = ut.load_datasets(max_samples=None) # Usando conjunto de datos completo para evaluación final
    
    # 1.5. Validar y filtrar señales muy cortas
    print(f"\n{'─' * 70}")
    print("Validando longitud de señales...")
    X_train, y_train, train_removed = ut.filter_valid_signals(
        X_train, y_train, 
        dim=BEST_PARAMS['dim'], 
        tau=BEST_PARAMS['tau'],
        verbose=True
    )
    X_test, y_test, test_removed = ut.filter_valid_signals(
        X_test, y_test,
        dim=BEST_PARAMS['dim'],
        tau=BEST_PARAMS['tau'],
        verbose=True
    )
    
    if train_removed > 0 or test_removed > 0:
        print(f"\nSeñales válidas después del filtrado:")
        print(f"  Entrenamiento: {len(X_train)} señales")
        print(f"  Prueba: {len(X_test)} señales")
        print(f"  Etiquetas entrenamiento: {np.bincount(y_train)}")
        print(f"  Etiquetas prueba: {np.bincount(y_test)}")
    
    # 2. Inicializar Modelo
    print(f"\nInicializando BinaryClassificationTE con parámetros óptimos...")
    model = BinaryClassificationTE(**BEST_PARAMS, seed=MODEL_SEED)
    
    # 3. Entrenar
    print(f"\n{'─' * 70}")
    print("Entrenando Modelo...")
    start_time = time.time()
    model.fit(X_train, y_train, verbose=True)
    train_time = time.time() - start_time
    print(f"✓ Entrenamiento completado en {train_time:.2f} segundos")
    
    # 4. Evaluación
    print(f"\n{'─' * 70}")
    print("Evaluando en Datos de Prueba...")
    start_time = time.time()
    
    # Obtener predicciones y probabilidades
    y_pred = model.predict(X_test)
    y_proba_all = model.predict_proba(X_test)
    y_proba = y_proba_all[:, 1]  # Probabilidad de clase positiva
    
    eval_time = time.time() - start_time
    print(f"✓ Predicción completada en {eval_time:.2f} segundos")
    
    # 5. Cálculo de Métricas
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=['Ruido', 'Terremoto'])
    
    # Imprimir Métricas
    print(f"\n{'=' * 70}")
    print("RESULTADOS FINALES")
    print(f"{'=' * 70}")
    print(f"Exactitud:  {acc:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print(f"\nReporte de Clasificación:\n{report}")
    
    # 6. Visualizaciones
    print(f"\n{'=' * 70}")
    print("GENERANDO VISUALIZACIONES")
    print(f"{'=' * 70}")
    
    plot_confusion_matrix(y_test, y_pred, RESULTS_DIR / "confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, auc, RESULTS_DIR / "roc_curve.png")
    plot_examples_grid(
        X_test,
        y_test,
        y_pred,
        y_proba,
        samples_per_class=3,
        model=model,
        model_params=BEST_PARAMS,
        save_path=RESULTS_DIR / "examples_grid.png",
        seed=42,  # Different seed for varied examples
        min_h0_points=10,
    )
    
    # 7. Análisis de Errores
    analyze_errors(X_test, y_test, y_pred, y_proba)
    
    print(f"\n{'=' * 70}")
    print(f"✓ Evaluación Completada. Resultados guardados en: {RESULTS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
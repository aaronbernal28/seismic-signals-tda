"""
Script: 07_diagnose_empty_diagrams.py
Descripción: Diagnosticar por qué algunas señales producen diagramas de persistencia vacíos o casi vacíos.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import RESULTS_DIR, MAX_SAMPLES
from config.best_te_params import BEST_PARAMS, MODEL_SEED
from src.models.model2 import BinaryClassificationTE
import src.utils as ut

def analyze_diagram_sizes(X, y, model_params):
    """Analizar tamaños de diagramas de persistencia para todas las señales."""
    
    print("=" * 80)
    print("ANÁLISIS DE TAMAÑOS DE DIAGRAMAS DE PERSISTENCIA")
    print("=" * 80)
    
    # Crear instancia temporal del modelo solo para usar transform
    temp_model = BinaryClassificationTE(**model_params, seed=MODEL_SEED)
    
    # Almacenar resultados
    results = {
        'index': [],
        'label': [],
        'signal_length': [],
        'signal_mean': [],
        'signal_std': [],
        'signal_min': [],
        'signal_max': [],
        'dgm_h0_size': [],
        'dgm_h1_size': [],
        'embedding_size': [],
    }
    
    print(f"\nProcesando {len(X)} señales...")
    
    for idx, (signal, label) in enumerate(zip(X, y)):
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
            
            # Filtrar puntos infinitos
            dgm_h0 = dgms[0][dgms[0][:, 1] != np.inf] if len(dgms[0]) > 0 else dgms[0]
            dgm_h1 = dgms[1][dgms[1][:, 1] != np.inf] if len(dgms[1]) > 0 else dgms[1]
            
            # Guardar resultados
            results['index'].append(idx)
            results['label'].append(label)
            results['signal_length'].append(len(signal))
            results['signal_mean'].append(np.mean(signal))
            results['signal_std'].append(np.std(signal))
            results['signal_min'].append(np.min(signal))
            results['signal_max'].append(np.max(signal))
            results['dgm_h0_size'].append(len(dgm_h0))
            results['dgm_h1_size'].append(len(dgm_h1))
            results['embedding_size'].append(len(embedding))
            
        except Exception as e:
            print(f"Error procesando señal {idx}: {e}")
            continue
    
    return results

def print_statistics(results):
    """Imprimir estadísticas sobre los tamaños de diagramas."""
    
    dgm_h0_sizes = np.array(results['dgm_h0_size'])
    dgm_h1_sizes = np.array(results['dgm_h1_size'])
    labels = np.array(results['label'])
    
    print("\n" + "=" * 80)
    print("ESTADÍSTICAS DE TAMAÑOS DE DIAGRAMAS")
    print("=" * 80)
    
    print("\n--- H0 (Componentes Conectadas) ---")
    print(f"Total de señales: {len(dgm_h0_sizes)}")
    print(f"Tamaño medio: {np.mean(dgm_h0_sizes):.2f} ± {np.std(dgm_h0_sizes):.2f}")
    print(f"Mínimo: {np.min(dgm_h0_sizes)}, Máximo: {np.max(dgm_h0_sizes)}")
    print(f"Diagramas vacíos (tamaño 0): {np.sum(dgm_h0_sizes == 0)}")
    print(f"Diagramas pequeños (tamaño 1-5): {np.sum((dgm_h0_sizes > 0) & (dgm_h0_sizes <= 5))}")
    
    print("\n--- H1 (Ciclos/Loops) ---")
    print(f"Total de señales: {len(dgm_h1_sizes)}")
    print(f"Tamaño medio: {np.mean(dgm_h1_sizes):.2f} ± {np.std(dgm_h1_sizes):.2f}")
    print(f"Mínimo: {np.min(dgm_h1_sizes)}, Máximo: {np.max(dgm_h1_sizes)}")
    print(f"Diagramas vacíos (tamaño 0): {np.sum(dgm_h1_sizes == 0)}")
    print(f"Diagramas pequeños (tamaño 1-5): {np.sum((dgm_h1_sizes > 0) & (dgm_h1_sizes <= 5))}")
    
    print("\n--- Por Clase ---")
    for label in [0, 1]:
        label_name = "Ruido" if label == 0 else "Sismo"
        mask = labels == label
        print(f"\n{label_name} (Clase {label}):")
        print(f"  H0 vacíos: {np.sum((dgm_h0_sizes[mask] == 0))} de {np.sum(mask)}")
        print(f"  H1 vacíos: {np.sum((dgm_h1_sizes[mask] == 0))} de {np.sum(mask)}")
        print(f"  H0 medio: {np.mean(dgm_h0_sizes[mask]):.2f}")
        print(f"  H1 medio: {np.mean(dgm_h1_sizes[mask]):.2f}")

def find_problematic_signals(results, top_n=10):
    """Identificar señales con diagramas vacíos o problemáticos."""
    
    print("\n" + "=" * 80)
    print("SEÑALES PROBLEMÁTICAS")
    print("=" * 80)
    
    dgm_h1_sizes = np.array(results['dgm_h1_size'])
    indices = np.array(results['index'])
    
    # Encontrar señales con H1 vacío
    empty_h1_indices = indices[dgm_h1_sizes == 0]
    
    if len(empty_h1_indices) > 0:
        print(f"\n--- Señales con H1 vacío ({len(empty_h1_indices)} encontradas) ---")
        print("\nPrimeras {} señales:".format(min(top_n, len(empty_h1_indices))))
        
        for i, idx in enumerate(empty_h1_indices[:top_n]):
            result_idx = np.where(indices == idx)[0][0]
            label_name = "Ruido" if results['label'][result_idx] == 0 else "Sismo"
            print(f"\nSeñal {idx} ({label_name}):")
            print(f"  Longitud: {results['signal_length'][result_idx]}")
            print(f"  Embedding size: {results['embedding_size'][result_idx]}")
            print(f"  Media: {results['signal_mean'][result_idx]:.2e}")
            print(f"  Desv. Est: {results['signal_std'][result_idx]:.2e}")
            print(f"  Rango: [{results['signal_min'][result_idx]:.2e}, {results['signal_max'][result_idx]:.2e}]")
            print(f"  H0 tamaño: {results['dgm_h0_size'][result_idx]}")
            print(f"  H1 tamaño: {results['dgm_h1_size'][result_idx]}")
    
    # Encontrar señales con H1 muy pequeño (1-3 puntos)
    small_h1_mask = (dgm_h1_sizes > 0) & (dgm_h1_sizes <= 3)
    small_h1_indices = indices[small_h1_mask]
    
    if len(small_h1_indices) > 0:
        print(f"\n--- Señales con H1 pequeño (1-3 puntos): {len(small_h1_indices)} encontradas ---")
        print("\nPrimeras {} señales:".format(min(top_n, len(small_h1_indices))))
        
        for i, idx in enumerate(small_h1_indices[:top_n]):
            result_idx = np.where(indices == idx)[0][0]
            label_name = "Ruido" if results['label'][result_idx] == 0 else "Sismo"
            print(f"\nSeñal {idx} ({label_name}):")
            print(f"  H0 tamaño: {results['dgm_h0_size'][result_idx]}")
            print(f"  H1 tamaño: {results['dgm_h1_size'][result_idx]}")
            print(f"  Desv. Est: {results['signal_std'][result_idx]:.2e}")

def plot_signal_properties(results, save_path):
    """Crear gráficos de distribución de propiedades de señales."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    dgm_h0_sizes = np.array(results['dgm_h0_size'])
    dgm_h1_sizes = np.array(results['dgm_h1_size'])
    labels = np.array(results['label'])
    signal_stds = np.array(results['signal_std'])
    signal_lengths = np.array(results['signal_length'])
    embedding_sizes = np.array(results['embedding_size'])
    
    # Plot 1: Distribución de tamaños H0
    axes[0, 0].hist([dgm_h0_sizes[labels == 0], dgm_h0_sizes[labels == 1]], 
                     bins=20, label=['Ruido', 'Sismo'], alpha=0.7)
    axes[0, 0].set_xlabel('Tamaño H0')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].set_title('Distribución Tamaño H0')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Distribución de tamaños H1
    axes[0, 1].hist([dgm_h1_sizes[labels == 0], dgm_h1_sizes[labels == 1]], 
                     bins=20, label=['Ruido', 'Sismo'], alpha=0.7)
    axes[0, 1].set_xlabel('Tamaño H1')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución Tamaño H1')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: H1 vs Desviación Estándar
    for label in [0, 1]:
        mask = labels == label
        label_name = "Ruido" if label == 0 else "Sismo"
        axes[0, 2].scatter(signal_stds[mask], dgm_h1_sizes[mask], 
                          alpha=0.5, s=20, label=label_name)
    axes[0, 2].set_xlabel('Desviación Estándar de Señal')
    axes[0, 2].set_ylabel('Tamaño H1')
    axes[0, 2].set_title('H1 vs Desviación Estándar')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Longitud de señal
    axes[1, 0].hist([signal_lengths[labels == 0], signal_lengths[labels == 1]], 
                     bins=20, label=['Ruido', 'Sismo'], alpha=0.7)
    axes[1, 0].set_xlabel('Longitud de Señal')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución Longitud de Señal')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Tamaño de embedding
    axes[1, 1].hist([embedding_sizes[labels == 0], embedding_sizes[labels == 1]], 
                     bins=20, label=['Ruido', 'Sismo'], alpha=0.7)
    axes[1, 1].set_xlabel('Tamaño de Embedding')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title('Distribución Tamaño de Embedding')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: H0 vs H1
    for label in [0, 1]:
        mask = labels == label
        label_name = "Ruido" if label == 0 else "Sismo"
        axes[1, 2].scatter(dgm_h0_sizes[mask], dgm_h1_sizes[mask], 
                          alpha=0.5, s=20, label=label_name)
    axes[1, 2].set_xlabel('Tamaño H0')
    axes[1, 2].set_ylabel('Tamaño H1')
    axes[1, 2].set_title('H0 vs H1')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n✓ Gráficos guardados en {save_path}")


def plot_prob_histograms(model, X, y, save_path, title):
    """Graficar histogramas de P(Sismo) por clase (dos colores)."""
    probs = model.predict_proba(X)
    p_terr = probs[:, 1]

    p_terr_noise = p_terr[y == 0]
    p_terr_quake = p_terr[y == 1]

    plt.figure(figsize=(7, 4))
    sns.histplot(p_terr_noise, color='steelblue', alpha=0.6, bins=20, label='Ruido (y=0)')
    sns.histplot(p_terr_quake, color='darkorange', alpha=0.6, bins=20, label='Sismo (y=1)')
    plt.xlabel('P(Sismo)')
    plt.ylabel('Frecuencia')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Histograma de probabilidades guardado en {save_path}")

def main():
    """Flujo principal."""
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(MODEL_SEED)
    
    # Cargar datos
    print("Cargando datos de prueba...")
    X_train, y_train, X_test, y_test = ut.load_datasets(max_samples=MAX_SAMPLES)
    
    # Filtrar señales válidas (mismo criterio que el modelo principal)
    X_train, y_train, _ = ut.filter_valid_signals(
        X_train, y_train,
        dim=BEST_PARAMS['dim'],
        tau=BEST_PARAMS['tau'],
        verbose=False
    )
    X_test, y_test, _ = ut.filter_valid_signals(
        X_test, y_test,
        dim=BEST_PARAMS['dim'],
        tau=BEST_PARAMS['tau'],
        verbose=False
    )
    
    print(f"Total de señales a analizar: {len(X_test)}")
    
    # Analizar tamaños de diagramas
    results = analyze_diagram_sizes(X_test, y_test, BEST_PARAMS)
    
    # Imprimir estadísticas
    print_statistics(results)
    
    # Encontrar señales problemáticas
    find_problematic_signals(results, top_n=10)
    
    # Crear gráficos
    plot_signal_properties(results, RESULTS_DIR / "07_diagram_sizes_analysis.png")
    
    # Histogramas de probabilidades en train y test
    print("Calculando probabilidades en train y test...")
    temp_model = BinaryClassificationTE(**BEST_PARAMS, seed=MODEL_SEED)
    temp_model.fit(X_train, y_train, verbose=False)
    plot_prob_histograms(temp_model, X_train, y_train, RESULTS_DIR / "07_train_prob_hist.png", "Distribución de P(Sismo) en Train")
    plot_prob_histograms(temp_model, X_test, y_test, RESULTS_DIR / "07_test_prob_hist.png", "Distribución de P(Sismo) en Test")
    
    print("\n" + "=" * 80)
    print("✓ Análisis completado")
    print("=" * 80)

if __name__ == "__main__":
    main()

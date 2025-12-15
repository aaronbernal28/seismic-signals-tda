"""
Búsqueda aleatoria en cuadrícula con validación cruzada para el modelo BinaryClassificationTE.
Métrica principal: ROC AUC
Usa sklearn.model_selection.RandomizedSearchCV
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.models.model2 import BinaryClassificationTE
import src.utils as ut
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
from persim import wasserstein, bottleneck
from src.cache import get_cache

def randomized_search_te(X, y, param_distributions, n_iter=10, cv=3, random_state=28, n_jobs=1):
    """Búsqueda aleatoria para BinaryClassificationTE usando RandomizedSearchCV de sklearn."""
    print("=" * 80)
    print("BÚSQUEDA ALEATORIA EN CUADRÍCULA: BinaryClassificationTE")
    print("=" * 80)
    print(f"Parámetros a explorar: {list(param_distributions.keys())}")
    print(f"Número de iteraciones: {n_iter}")
    print(f"Pliegues de validación cruzada: {cv}")
    print("=" * 80)
    
    # Inicializar modelo base
    base_model = BinaryClassificationTE()
    
    # Crear scorer para ROC AUC
    auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
    
    # Crear RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=auc_scorer,
        random_state=random_state,
        verbose=2,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score=np.nan
    )
    
    # Ajustar la búsqueda aleatoria
    print("\nIniciando búsqueda aleatoria...\n")
    random_search.fit(X, y)
    
    return random_search


def display_results(search, model_name):
    """Mostrar y formatear resultados de RandomizedSearchCV."""
    # Convertir resultados a DataFrame
    results_df = pd.DataFrame(search.cv_results_)
    
    # Seleccionar columnas relevantes
    columns_to_keep = [col for col in results_df.columns if col.startswith('param_') or 
                       col in ['mean_test_score', 'std_test_score', 'rank_test_score', 
                               'mean_train_score', 'std_train_score', 'mean_fit_time']]
    results_df = results_df[columns_to_keep]
    
    # Renombrar columnas para claridad
    results_df = results_df.rename(columns={
        'mean_test_score': 'mean_auc',
        'std_test_score': 'std_auc',
        'mean_train_score': 'mean_train_auc',
        'std_train_score': 'std_train_auc',
        'mean_fit_time': 'fit_time',
        'rank_test_score': 'rank'
    })
    
    # Quitar el prefijo 'param_' de las columnas de parámetros
    results_df.columns = [col.replace('param_', '') if col.startswith('param_') else col 
                          for col in results_df.columns]
    
    # Ordenar por AUC promedio
    results_df = results_df.sort_values('rank').reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print(f"RESULTADOS: {model_name}")
    print("=" * 80)
    print(f"\nMejores parámetros: {search.best_params_}")
    print(f"Mejor AUC: {search.best_score_:.4f}")
    print("\nTop 5 configuraciones:")
    print(results_df.head(5).to_string())
    
    return results_df


def main():
    """Función principal de ejecución."""
    print("=" * 80)
    print("BÚSQUEDA ALEATORIA EN CUADRÍCULA CON VALIDACIÓN CRUZADA")
    print("Métrica principal: ROC AUC")
    print("=" * 80)
    
    # Fijar semilla
    np.random.seed(28)
    
    # Cargar datasets
    X_train, y_train, _, _ = ut.load_datasets(max_samples=100)
    
    # =========================================================================
    # DISTRIBUCIONES DE HIPERPARÁMETROS - COMPLETAR SI ES NECESARIO
    # =========================================================================
    
    # Distribuciones de parámetros para BinaryClassificationTE (Takens Embedding)
    param_distributions = {
        'distance': [wasserstein],  # métrica de distancia entre diagramas de persistencia
        'weights': [(1,), (1, 1), (2, 1), (1, 2)],  # pesos para el cálculo de distancia (tuplas para compatibilidad)
        'sample': list(range(10, 20)),  # cantidad de diagramas a muestrear
        'thresh': np.linspace(1000, 3000, 10),  # umbral
        'alpha': np.linspace(0.25, 0.9, 10),  # proporción de subsampling FPS
        'max_points': list(range(100, 300, 50)),  # número máximo de puntos de Takens
        'seed': [28]  # semilla para reproducibilidad
    }

    # Distribuciones de parámetros específicas para BinaryClassificationTE (Takens Embedding)
    param_distributions_te = {
        **param_distributions,
        'dim': [int(2**i) for i in range(1, 5)],  # dimensión de la incrustación de Takens
        'tau': [int(2**i) for i in range(1, 5)],  # retardo temporal
        'stride': list(range(1, 5)),  # stride
    }
    
    # =========================================================================
    # PARÁMETROS DE LA BÚSQUEDA
    # =========================================================================
    n_iter = 100  # número de combinaciones aleatorias por modelo
    cv_folds = 3  # número de pliegues de validación cruzada
    random_state = 28  # semilla para reproducibilidad
    n_jobs = 10  # número de trabajos paralelos para RandomizedSearchCV
    
    # =========================================================================
    # EJECUTAR BÚSQUEDA ALEATORIA PARA EL MODELO 2 (Takens Embedding)
    # =========================================================================
    print("\n")
    search_te = randomized_search_te(
        X_train, y_train, 
        param_distributions=param_distributions_te,
        n_iter=n_iter,
        cv=cv_folds,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    # Mostrar y guardar resultados
    results_te = display_results(search_te, "BinaryClassificationTE")
    
    output_path_te = Path("data/results/grid_search_te.csv")
    output_path_te.parent.mkdir(exist_ok=True)
    results_te.to_csv(output_path_te, index=False)
    print(f"\n✓ Resultados guardados en: {output_path_te}")
    
    # Imprimir estadísticas del caché luego de TE
    get_cache().print_stats()
    
    # =========================================================================
    # RESUMEN
    # =========================================================================
    print("\n" + "=" * 80)
    print("BÚSQUEDA ALEATORIA COMPLETADA")
    print("=" * 80)
    print(f"\nMejor BinaryClassificationTE:")
    print(f"  AUC: {search_te.best_score_:.4f}")
    print(f"  Parámetros: {search_te.best_params_}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

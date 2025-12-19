""" Implementación del modelo de clasificación binaria. """

import numpy as np
from src.databases import PersistenceDiagramDatabaseTE
from persim import bottleneck
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin


class BinaryClassificationTE(PersistenceDiagramDatabaseTE, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        distance=bottleneck,
        weights=(1,),
        dim=100,
        tau=10,
        stride=1,
        maxdim=None,
        sample=10,
        thresh=np.inf,
        alpha=0.1,
        max_points=500,
        seed=28,
        normalize_minmax=True,
    ):
        """Inicializar el modelo de clasificación binaria.
        Args:
            distance: función para calcular distancia entre diagramas de persistencia
            weights: secuencia de floats, pesos para cada dimensión de homología
            dim: int, dimensión de incrustación
            tau: int, retardo temporal
            stride: int, paso para ventana deslizante
            maxdim: int o None, dimensión máxima de homología a calcular (si None, inferida de weights)
            sample: int o None, número de diagramas a muestrear para cálculo de distancia
            thresh: float, umbral para puntos del diagrama
            alpha: float, proporción de subsampling FPS
            max_points: int o np.inf, límite para puntos FPS
            seed: int, semilla RNG
            normalize_minmax: bool, normalizar cada señal a [0,1] antes de Takens
        """
        # Almacenar parámetros exactamente como se dieron (compatibilidad con scikit-learn)
        self.distance = distance
        self.weights = weights
        self.dim = dim
        self.tau = tau
        self.stride = stride
        self.maxdim = maxdim
        self.sample = sample
        self.thresh = thresh
        self.alpha = alpha
        self.max_points = max_points
        self.seed = seed
        self.normalize_minmax = normalize_minmax

        # Inicialización lazy de la base de datos una vez que todos los parámetros efectivos se conocen
        self._initialized = False
        self._normalized_weights = None

    # Ayudantes internos -----------------------------------------------------
    def _effective_maxdim(self):
        return (len(self.weights) - 1) if self.maxdim is None else self.maxdim

    def _ensure_initialized(self):
        if not self._initialized:
            # Inicializar base de datos con maxdim efectivo
            PersistenceDiagramDatabaseTE.__init__(
                self,
                dim=self.dim,
                tau=self.tau,
                stride=self.stride,
                maxdim=self._effective_maxdim(),
                sample=self.sample,
                thresh=self.thresh,
                alpha=self.alpha,
                max_points=self.max_points,
                seed=self.seed,
                normalize_minmax=self.normalize_minmax,
            )
            w = np.asarray(self.weights, dtype=float)
            s = np.sum(w)
            if s == 0:
                # Si todos los pesos son 0, usar pesos iguales
                self._normalized_weights = np.ones_like(w) / len(w)
            else:
                self._normalized_weights = w / s
            self._initialized = True
    
    def __sklearn_is_fitted__(self):
        """Verificar si el modelo está ajustado (para compatibilidad con scikit-learn)."""
        return hasattr(self, 'classes_') and self._initialized
    
    def fit(self, X, y, verbose=False):
        """ Ajustar el modelo de clasificación binaria. 
        Args:
            X: List[ndarray] (n_samples, ~n_signals)
            y: List[int] de etiquetas (0 o 1)
        """
        # Reiniciar estado para prevenir fuga de datos entre pliegues CV
        self._initialized = False
        self._normalized_weights = None
        
        self._ensure_initialized()
        
        # Limpiar base de datos de ajuste previo (crítico para CV)
        self.db = {label: {d: [] for d in range(self.maxdim + 1)} for label in self.labels}
        
        # Almacenar clases para compatibilidad con scikit-learn
        self.classes_ = np.unique(y)
        
        # Contador de señales omitidas
        skipped_count = 0
        skipped_by_label = {}
        
        for i, (xs, yi) in enumerate(zip(X, y)):
            success = self.add_signal(xs, label=yi)
            if not success:
                skipped_count += 1
                skipped_by_label[yi] = skipped_by_label.get(yi, 0) + 1
            if verbose and (i % 10 == 0):
                print(f"Se agregó señal {i}/{len(y)}", end='\r')
        
        if skipped_count > 0:
            print(f"\n⚠ Señales omitidas durante entrenamiento: {skipped_count}")
            for label, count in skipped_by_label.items():
                label_name = 'Sismo' if label == 1 else 'Ruido'
                print(f"  - {label_name} (etiqueta {label}): {count} señal(es)")
        
        if verbose:
            for label in [0, 1]:
                for d in range(self.maxdim + 1):
                    pc_len = []
                    for diagrams in self.get_diagrams(label=label, dim=d):
                        pc_len.append(len(diagrams))
                    # Imprimir estadísticas
                    if pc_len:
                        print(f"Etiqueta {label}, Dim {d}: Tamaño medio del diagrama: {np.mean(pc_len):.2f}, Desv: {np.std(pc_len):.2f}, Máx: {np.max(pc_len)}, Mín: {np.min(pc_len)}")
        print("Ajuste del modelo completo.")
        return self

    def predict_proba_sample(self, xs):
        """ Predecir la probabilidad para una muestra única.
        Args:
            xs: ndarray (~n_signals,)
        Returns:
            prob: float en [0, 1]
        """
        self._ensure_initialized()
        # Calcular diagrama de persistencia para la señal de entrada
        try:
            xs_dgm = self.transform(xs)
        except Exception as e:
            print(f"Error transformando la señal de entrada: {e}")
            return 0.5  # Devolver una probabilidad neutral en caso de error

        mean_dist_E = []
        mean_dist_N = []

        for d in range(self.maxdim + 1):
            # Proceder solo si hay puntos en el diagrama
            if len(xs_dgm[d]) > 0:
                # Recuperar diagramas de la base de datos
                E_h0 = self.get_diagrams(label=1, dim=d)
                N_h0 = self.get_diagrams(label=0, dim=d)
                #print(f"Dimension {d}: {len(E_h0)} diagrams for class 1, {len(N_h0)} diagrams for class 0.", end='\r')
                # Calcular distancias
                dists_E = [self.distance(xs_dgm[d], dgm) for dgm in E_h0]
                dists_N = [self.distance(xs_dgm[d], dgm) for dgm in N_h0]

                mean_dist_E.append(np.mean(dists_E) if dists_E else np.inf) 
                mean_dist_N.append(np.mean(dists_N) if dists_N else np.inf)
            else:
                mean_dist_E.append(np.inf) 
                mean_dist_N.append(np.inf) 

        #prob = np.exp(mean_dist_E) / (np.exp(mean_dist_E) + np.exp(mean_dist_N))
        # Ponderar distancias con pesos entre dimensiones de homología
        mean_dist_E = np.sum([w * d for w, d in zip(self._normalized_weights, mean_dist_E)])
        mean_dist_N = np.sum([w * d for w, d in zip(self._normalized_weights, mean_dist_N)])

        # Solo para estar seguro
        if mean_dist_E == np.inf and mean_dist_N == np.inf:
            prob = 0.5
        elif mean_dist_E == np.inf and mean_dist_N < np.inf:
            prob = 0.0
        elif mean_dist_E < np.inf and mean_dist_N == np.inf:
            prob = 1.0
        else:
            prob = 1 / (1 + np.exp(mean_dist_E - mean_dist_N))
        return prob
    
    def predict_proba(self, X):
        """ Predecir probabilidades para múltiples muestras.
        Args:
            X: List[ndarray] (n_samples, ~n_signals)
        Returns:
            probs: ndarray de forma (n_samples, 2) con probabilidades para cada clase
        """
        self._ensure_initialized()
        probs_class1 = [self.predict_proba_sample(xs) for xs in X]
        # Devolver arreglo 2D: [prob_clase_0, prob_clase_1] para cada muestra
        probs = np.array([[1 - p, p] for p in probs_class1])
        return probs

    def predict(self, X):
        """ Predecir etiquetas de clase para múltiples muestras.
        Args:
            X: List[ndarray] (n_samples, ~n_signals)
        Returns:
            labels: ndarray de etiquetas predichas (0 o 1)
        """
        probs = self.predict_proba(X)
        # Usar probabilidad de clase 1 (segunda columna)
        labels = (probs[:, 1] >= 0.5).astype(int)
        return labels

    def score(self, X, y):
        """ Calcular el puntaje ROC AUC.
        Args:
            X: List[ndarray] (n_samples, ~n_signals)
            y: List[int] de etiquetas verdaderas (0 o 1)
        Returns:
            auc: float puntaje ROC AUC
        """
        probs = self.predict_proba(X)
        # Usar probabilidades para clase 1 (clase positiva)
        auc = roc_auc_score(y, probs[:, 1])
        return auc
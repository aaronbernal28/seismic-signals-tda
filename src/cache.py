"""
Caché de diagramas de persistencia para señales sísmicas.
Almacena resultados de las transformaciones (diagramas) indexados por el contenido de la señal y los hiperparámetros.
Seguro para hilos y para validación cruzada (no filtra datos ajustados entre pliegues).
"""
import hashlib
import pickle
from typing import Tuple, Optional, Dict, Any
import numpy as np
from threading import Lock


class PersistenceDiagramCache:
    """Caché de diagramas de persistencia calculados a partir de señales.
    
    Diseño clave:
    - Clave del caché = hash(datos_de_señal + parámetros_de_transformación)
    - Solo guarda resultados de transformaciones, NO bases de datos ajustadas
    - Seguro para hilos durante CV paralelo
    - Cachés separados para TE y MFCC para evitar colisiones
    """
    
    def __init__(self):
        self._cache_te: Dict[str, list] = {}
        self._cache_mfcc: Dict[str, list] = {}
        self._lock = Lock()
        self._hits_te = 0
        self._misses_te = 0
        self._hits_mfcc = 0
        self._misses_mfcc = 0
    
    def _hash_signal(self, signal: np.ndarray) -> str:
        """Crear un hash a partir de los datos de la señal."""
        # Usar un subconjunto si la señal es muy grande para acelerar el hashing
        if len(signal) > 10000:
            # Hashear los primeros 5000, los últimos 5000 y la longitud
            data = np.concatenate([signal[:5000], signal[-5000:], [len(signal)]])
        else:
            data = signal
        return hashlib.blake2b(data.tobytes(), digest_size=16).hexdigest()
    
    def _hash_params(self, params: Dict[str, Any]) -> str:
        """Crear un hash a partir del diccionario de parámetros."""
        # Ordenar claves para mantener consistencia
        sorted_items = sorted(params.items())
        param_str = str(sorted_items)
        return hashlib.blake2b(param_str.encode(), digest_size=16).hexdigest()
    
    def _make_key(self, signal: np.ndarray, params: Dict[str, Any]) -> str:
        """Crear la clave del caché a partir de la señal y los parámetros."""
        signal_hash = self._hash_signal(signal)
        param_hash = self._hash_params(params)
        return f"{signal_hash}_{param_hash}"
    
    def get_te(self, signal: np.ndarray, dim: int, tau: int, stride: int, 
               maxdim: int, thresh: float, alpha: float, max_points: int) -> Optional[list]:
        """Obtener diagramas TE en caché si están disponibles.
        
        Args:
            signal: Señal de entrada
            dim: Dimensión de la incrustación de Takens
            tau: Retardo temporal
            stride: Paso para la incrustación de Takens
            maxdim: Dimensión máxima de homología
            thresh: Umbral de Vietoris-Rips
            alpha: Proporción de subsampling FPS
            max_points: Número máximo de puntos después del subsampling
            
        Returns:
            Diagramas en caché o None si no se encuentran
        """
        params = {
            'dim': dim, 'tau': tau, 'stride': stride,
            'maxdim': maxdim, 'thresh': thresh, 
            'alpha': alpha, 'max_points': max_points
        }
        key = self._make_key(signal, params)
        
        with self._lock:
            if key in self._cache_te:
                self._hits_te += 1
                # Devolver una copia profunda para evitar mutaciones
                return [dgm.copy() for dgm in self._cache_te[key]]
            else:
                self._misses_te += 1
                return None
    
    def put_te(self, signal: np.ndarray, dim: int, tau: int, stride: int,
               maxdim: int, thresh: float, alpha: float, max_points: int, 
               diagrams: list) -> None:
        """Almacenar diagramas TE en el caché.
        
        Args:
            signal: Señal de entrada
            dim: Dimensión de incrustación de Takens
            tau: Retardo temporal
            stride: Paso para la incrustación de Takens
            maxdim: Dimensión máxima de homología
            thresh: Umbral de Vietoris-Rips
            alpha: Proporción de subsampling FPS
            max_points: Máximo de puntos tras el subsampling
            diagrams: Lista de diagramas de persistencia a guardar
        """
        params = {
            'dim': dim, 'tau': tau, 'stride': stride,
            'maxdim': maxdim, 'thresh': thresh,
            'alpha': alpha, 'max_points': max_points
        }
        key = self._make_key(signal, params)
        
        with self._lock:
            # Guardar una copia profunda para prevenir mutaciones
            self._cache_te[key] = [dgm.copy() for dgm in diagrams]
    
    def get_mfcc(self, signal: np.ndarray, n_mfcc: int, sr: float, 
                 win_length: int, hop_length: int, maxdim: int, 
                 thresh: float, alpha: float, max_points: int) -> Optional[list]:
        """Obtener diagramas MFCC en caché si están disponibles.
        
        Args:
            signal: Señal de entrada
            n_mfcc: Cantidad de coeficientes MFCC
            sr: Frecuencia de muestreo
            win_length: Longitud de ventana para MFCC
            hop_length: Paso entre ventanas
            maxdim: Dimensión máxima de homología
            thresh: Umbral de Vietoris-Rips
            alpha: Proporción de subsampling FPS
            max_points: Máximo de puntos tras el subsampling
            
        Returns:
            Diagramas en caché o None si no se encuentran
        """
        params = {
            'n_mfcc': n_mfcc, 'sr': sr, 'win_length': win_length,
            'hop_length': hop_length, 'maxdim': maxdim, 'thresh': thresh,
            'alpha': alpha, 'max_points': max_points
        }
        key = self._make_key(signal, params)
        
        with self._lock:
            if key in self._cache_mfcc:
                self._hits_mfcc += 1
                # Devolver una copia profunda para evitar mutaciones
                return [dgm.copy() for dgm in self._cache_mfcc[key]]
            else:
                self._misses_mfcc += 1
                return None
    
    def put_mfcc(self, signal: np.ndarray, n_mfcc: int, sr: float,
                 win_length: int, hop_length: int, maxdim: int,
                 thresh: float, alpha: float, max_points: int,
                 diagrams: list) -> None:
        """Guardar diagramas MFCC en el caché.
        
        Args:
            signal: Señal de entrada
            n_mfcc: Cantidad de coeficientes MFCC
            sr: Frecuencia de muestreo
            win_length: Longitud de ventana para MFCC
            hop_length: Paso entre ventanas
            maxdim: Dimensión máxima de homología
            thresh: Umbral de Vietoris-Rips
            alpha: Proporción de subsampling FPS
            max_points: Máximo de puntos tras el subsampling
            diagrams: Lista de diagramas de persistencia a guardar
        """
        params = {
            'n_mfcc': n_mfcc, 'sr': sr, 'win_length': win_length,
            'hop_length': hop_length, 'maxdim': maxdim, 'thresh': thresh,
            'alpha': alpha, 'max_points': max_points
        }
        key = self._make_key(signal, params)
        
        with self._lock:
            # Guardar una copia profunda para prevenir mutaciones
            self._cache_mfcc[key] = [dgm.copy() for dgm in diagrams]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché."""
        with self._lock:
            total_te = self._hits_te + self._misses_te
            total_mfcc = self._hits_mfcc + self._misses_mfcc
            
            hit_rate_te = (self._hits_te / total_te * 100) if total_te > 0 else 0
            hit_rate_mfcc = (self._hits_mfcc / total_mfcc * 100) if total_mfcc > 0 else 0
            
            return {
                'te': {
                    'hits': self._hits_te,
                    'misses': self._misses_te,
                    'total': total_te,
                    'hit_rate': hit_rate_te,
                    'cache_size': len(self._cache_te)
                },
                'mfcc': {
                    'hits': self._hits_mfcc,
                    'misses': self._misses_mfcc,
                    'total': total_mfcc,
                    'hit_rate': hit_rate_mfcc,
                    'cache_size': len(self._cache_mfcc)
                }
            }
    
    def clear(self) -> None:
        """Vaciar todos los cachés y reiniciar estadísticas."""
        with self._lock:
            self._cache_te.clear()
            self._cache_mfcc.clear()
            self._hits_te = 0
            self._misses_te = 0
            self._hits_mfcc = 0
            self._misses_mfcc = 0
    
    def print_stats(self) -> None:
        """Imprimir estadísticas del caché."""
        stats = self.get_stats()
        print("\n" + "=" * 80)
        print("ESTADÍSTICAS DEL CACHÉ DE DIAGRAMAS DE PERSISTENCIA")
        print("=" * 80)
        
        print("\nTakens Embedding (TE):")
        print(f"  Tamaño del caché: {stats['te']['cache_size']} entradas")
        print(f"  Aciertos: {stats['te']['hits']}")
        print(f"  Fallos: {stats['te']['misses']}")
        print(f"  Solicitudes totales: {stats['te']['total']}")
        print(f"  Tasa de aciertos: {stats['te']['hit_rate']:.2f}%")
        
        print("\nMFCC:")
        print(f"  Tamaño del caché: {stats['mfcc']['cache_size']} entradas")
        print(f"  Aciertos: {stats['mfcc']['hits']}")
        print(f"  Fallos: {stats['mfcc']['misses']}")
        print(f"  Solicitudes totales: {stats['mfcc']['total']}")
        print(f"  Tasa de aciertos: {stats['mfcc']['hit_rate']:.2f}%")
        
        print("=" * 80)


# Global cache instance (thread-safe, shared across all models during grid search)
_global_cache = PersistenceDiagramCache()


def get_cache() -> PersistenceDiagramCache:
    """Obtener el caché global de diagramas de persistencia."""
    return _global_cache

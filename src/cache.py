"""
Persistence diagram cache for seismic signals.
Caches transform results (persistence diagrams) keyed by signal content and hyperparameters.
Thread-safe and safe for cross-validation (doesn't leak fitted data between folds).
"""
import hashlib
import pickle
from typing import Tuple, Optional, Dict, Any
import numpy as np
from threading import Lock


class PersistenceDiagramCache:
    """Cache for persistence diagrams computed from signals.
    
    Key design:
    - Cache key = hash(signal_data + transform_params)
    - Only caches transform results, NOT fitted databases
    - Thread-safe for parallel CV
    - Separate caches for TE and MFCC to avoid collisions
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
        """Create a hash from signal data."""
        # Use a subset if signal is huge to speed up hashing
        if len(signal) > 10000:
            # Hash first 5000, last 5000, and length
            data = np.concatenate([signal[:5000], signal[-5000:], [len(signal)]])
        else:
            data = signal
        return hashlib.blake2b(data.tobytes(), digest_size=16).hexdigest()
    
    def _hash_params(self, params: Dict[str, Any]) -> str:
        """Create a hash from parameter dictionary."""
        # Sort keys for consistency
        sorted_items = sorted(params.items())
        param_str = str(sorted_items)
        return hashlib.blake2b(param_str.encode(), digest_size=16).hexdigest()
    
    def _make_key(self, signal: np.ndarray, params: Dict[str, Any]) -> str:
        """Create cache key from signal and parameters."""
        signal_hash = self._hash_signal(signal)
        param_hash = self._hash_params(params)
        return f"{signal_hash}_{param_hash}"
    
    def get_te(self, signal: np.ndarray, dim: int, tau: int, stride: int, 
               maxdim: int, thresh: float, alpha: float, max_points: int) -> Optional[list]:
        """Get cached TE persistence diagrams if available.
        
        Args:
            signal: Input signal
            dim: Takens embedding dimension
            tau: Time delay
            stride: Stride for Takens embedding
            maxdim: Maximum homology dimension
            thresh: Vietoris-Rips threshold
            alpha: FPS subsampling proportion
            max_points: Maximum points after subsampling
            
        Returns:
            Cached diagrams or None if not in cache
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
                # Return a deep copy to prevent mutation
                return [dgm.copy() for dgm in self._cache_te[key]]
            else:
                self._misses_te += 1
                return None
    
    def put_te(self, signal: np.ndarray, dim: int, tau: int, stride: int,
               maxdim: int, thresh: float, alpha: float, max_points: int, 
               diagrams: list) -> None:
        """Store TE persistence diagrams in cache.
        
        Args:
            signal: Input signal
            dim: Takens embedding dimension
            tau: Time delay
            stride: Stride for Takens embedding
            maxdim: Maximum homology dimension
            thresh: Vietoris-Rips threshold
            alpha: FPS subsampling proportion
            max_points: Maximum points after subsampling
            diagrams: List of persistence diagrams to cache
        """
        params = {
            'dim': dim, 'tau': tau, 'stride': stride,
            'maxdim': maxdim, 'thresh': thresh,
            'alpha': alpha, 'max_points': max_points
        }
        key = self._make_key(signal, params)
        
        with self._lock:
            # Store a deep copy to prevent mutation
            self._cache_te[key] = [dgm.copy() for dgm in diagrams]
    
    def get_mfcc(self, signal: np.ndarray, n_mfcc: int, sr: float, 
                 win_length: int, hop_length: int, maxdim: int, 
                 thresh: float, alpha: float, max_points: int) -> Optional[list]:
        """Get cached MFCC persistence diagrams if available.
        
        Args:
            signal: Input signal
            n_mfcc: Number of MFCC coefficients
            sr: Sample rate
            win_length: Window length for MFCC
            hop_length: Hop length for MFCC
            maxdim: Maximum homology dimension
            thresh: Vietoris-Rips threshold
            alpha: FPS subsampling proportion
            max_points: Maximum points after subsampling
            
        Returns:
            Cached diagrams or None if not in cache
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
                # Return a deep copy to prevent mutation
                return [dgm.copy() for dgm in self._cache_mfcc[key]]
            else:
                self._misses_mfcc += 1
                return None
    
    def put_mfcc(self, signal: np.ndarray, n_mfcc: int, sr: float,
                 win_length: int, hop_length: int, maxdim: int,
                 thresh: float, alpha: float, max_points: int,
                 diagrams: list) -> None:
        """Store MFCC persistence diagrams in cache.
        
        Args:
            signal: Input signal
            n_mfcc: Number of MFCC coefficients
            sr: Sample rate
            win_length: Window length for MFCC
            hop_length: Hop length for MFCC
            maxdim: Maximum homology dimension
            thresh: Vietoris-Rips threshold
            alpha: FPS subsampling proportion
            max_points: Maximum points after subsampling
            diagrams: List of persistence diagrams to cache
        """
        params = {
            'n_mfcc': n_mfcc, 'sr': sr, 'win_length': win_length,
            'hop_length': hop_length, 'maxdim': maxdim, 'thresh': thresh,
            'alpha': alpha, 'max_points': max_points
        }
        key = self._make_key(signal, params)
        
        with self._lock:
            # Store a deep copy to prevent mutation
            self._cache_mfcc[key] = [dgm.copy() for dgm in diagrams]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
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
        """Clear all caches and reset statistics."""
        with self._lock:
            self._cache_te.clear()
            self._cache_mfcc.clear()
            self._hits_te = 0
            self._misses_te = 0
            self._hits_mfcc = 0
            self._misses_mfcc = 0
    
    def print_stats(self) -> None:
        """Print cache statistics."""
        stats = self.get_stats()
        print("\n" + "=" * 80)
        print("PERSISTENCE DIAGRAM CACHE STATISTICS")
        print("=" * 80)
        
        print("\nTakens Embedding (TE):")
        print(f"  Cache size: {stats['te']['cache_size']} entries")
        print(f"  Hits: {stats['te']['hits']}")
        print(f"  Misses: {stats['te']['misses']}")
        print(f"  Total requests: {stats['te']['total']}")
        print(f"  Hit rate: {stats['te']['hit_rate']:.2f}%")
        
        print("\nMFCC:")
        print(f"  Cache size: {stats['mfcc']['cache_size']} entries")
        print(f"  Hits: {stats['mfcc']['hits']}")
        print(f"  Misses: {stats['mfcc']['misses']}")
        print(f"  Total requests: {stats['mfcc']['total']}")
        print(f"  Hit rate: {stats['mfcc']['hit_rate']:.2f}%")
        
        print("=" * 80)


# Global cache instance (thread-safe, shared across all models during grid search)
_global_cache = PersistenceDiagramCache()


def get_cache() -> PersistenceDiagramCache:
    """Get the global persistence diagram cache."""
    return _global_cache

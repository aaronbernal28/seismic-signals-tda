""" Binary classification model implementation using MFCC features instead of Takens embeddings. """

import numpy as np
from src.databases import PersistenceDiagramDatabaseMFCC
from persim import bottleneck
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin


class BinaryClassificationMFCC(PersistenceDiagramDatabaseMFCC, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        distance=bottleneck,
        weights=(1,),
        n_mfcc=40,
        sr=40.0,
        win_length_sec=0.3,
        hop_length_sec=0.2,
        maxdim=None,
        sample=10,
        thresh=np.inf,
        alpha=0.1,
        max_points=500,
        seed=28,
    ):
        """Initialize the binary classification model using MFCC features.
        Stores params exactly for scikit-learn compatibility; lazily initializes the database.
        """
        self.distance = distance
        self.weights = weights
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.win_length_sec = win_length_sec
        self.hop_length_sec = hop_length_sec
        self.maxdim = maxdim
        self.sample = sample
        self.thresh = thresh
        self.alpha = alpha
        self.max_points = max_points
        self.seed = seed

        self._initialized = False
        self._normalized_weights = None

    # Internal helpers -----------------------------------------------------
    def _effective_maxdim(self):
        return (len(self.weights) - 1) if self.maxdim is None else self.maxdim

    def _ensure_initialized(self):
        if not self._initialized:
            PersistenceDiagramDatabaseMFCC.__init__(
                self,
                n_mfcc=self.n_mfcc,
                sr=self.sr,
                win_length_sec=self.win_length_sec,
                hop_length_sec=self.hop_length_sec,
                maxdim=self._effective_maxdim(),
                sample=self.sample,
                thresh=self.thresh,
                alpha=self.alpha,
                max_points=self.max_points,
                seed=self.seed,
            )
            w = np.asarray(self.weights, dtype=float)
            s = np.sum(w)
            self._normalized_weights = (w / s) if s != 0 else np.asarray(self.weights, dtype=float)
            self._initialized = True
    
    def fit(self, X, y, verbose=False):
        """ Fit the binary classification model. 
        Args:
            X: List[ndarray] - each element is a 1D signal
            y: List[int] of labels (0 or 1)
        """
        self._ensure_initialized()
        # Store classes for scikit-learn compatibility
        self.classes_ = np.unique(y)
        
        for i, (signal, yi) in enumerate(zip(X, y)):
            self.add_signal(signal, label=yi)
            if verbose and (i % 10 == 0):
                print(f"Added signal {i}/{len(y)}", end='\r')
        
        if verbose:
            for label in [0, 1]:
                for d in range(self.maxdim + 1):
                    pc_len = []
                    for diagrams in self.get_diagrams(label=label, dim=d):
                        pc_len.append(len(diagrams))
                    if pc_len:
                        print(f"Label {label}, Dim {d}: Mean diagram size: {np.mean(pc_len):.2f}, Std: {np.std(pc_len):.2f}, Max: {np.max(pc_len)}, Min: {np.min(pc_len)}")
        print("Model fitting complete.")
        return self

    def predict_proba_sample(self, signal):
        """ Predict the probability for a single sample.
        Args:
            signal: ndarray - 1D signal array
        Returns:
            prob: float in [0, 1]
        """
        self._ensure_initialized()
        try:
            xs_dgm = self.transform(signal)
        except Exception as e:
            print(f"Error transforming the input signal: {e}")
            return 0.5

        mean_dist_E = []
        mean_dist_N = []

        for d in range(self.maxdim + 1):
            if len(xs_dgm[d]) > 0:
                E_h0 = self.get_diagrams(label=1, dim=d)
                N_h0 = self.get_diagrams(label=0, dim=d)
                
                dists_E = [self.distance(xs_dgm[d], dgm) for dgm in E_h0]
                dists_N = [self.distance(xs_dgm[d], dgm) for dgm in N_h0]

                mean_dist_E.append(np.mean(dists_E) if dists_E else np.inf) 
                mean_dist_N.append(np.mean(dists_N) if dists_N else np.inf)
            else:
                mean_dist_E.append(np.inf) 
                mean_dist_N.append(np.inf) 

        # Weighted sum across homology dimensions
        mean_dist_E = np.sum([w * d for w, d in zip(self._normalized_weights, mean_dist_E)])
        mean_dist_N = np.sum([w * d for w, d in zip(self._normalized_weights, mean_dist_N)])

        # Handle edge cases
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
        """ Predict probabilities for multiple samples.
        Args:
            X: List[ndarray] - each element is a 1D signal
        Returns:
            probs: ndarray of shape (n_samples, 2) with probabilities for each class
        """
        self._ensure_initialized()
        probs_class1 = [self.predict_proba_sample(signal) for signal in X]
        # Return 2D array: [prob_class_0, prob_class_1] for each sample
        probs = np.array([[1 - p, p] for p in probs_class1])
        return probs

    def predict(self, X):
        """ Predict class labels for multiple samples.
        Args:
            X: List[ndarray] - each element is a 1D signal
        Returns:
            labels: List[int] of predicted labels (0 or 1)
        """
        probs = self.predict_proba(X)
        labels = [1 if p >= 0.5 else 0 for p in probs]
        return labels

    def score(self, X, y):
        """ Compute the ROC AUC score.
        Args:
            X: List[ndarray] - each element is a 1D signal
            y: List[int] of true labels (0 or 1)
        Returns:
            auc: float ROC AUC score
        """
        probs = self.predict_proba(X)
        # Use probabilities for class 1 (positive class)
        auc = roc_auc_score(y, probs[:, 1])
        return auc
    
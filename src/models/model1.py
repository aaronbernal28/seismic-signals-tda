""" Binary classification model implementation. 
For simplicity: 
    labels=[0, 1]
    maxdim=0    only H0 persistence diagrams
    sample=None     calculate the distance matrix against all diagrams
"""

import numpy as np
import pandas as pd
from src.databases import PersistenceDiagramDatabase
from src.utils import takens_embedding, compute_persistence
from persim import wasserstein, bottleneck
from sklearn.metrics import roc_auc_score

class BinaryClassificationModel:
    def __init__(self, dim=100, tau=10, distance=bottleneck, alpha=0.2):
        self.dim = dim
        self.tau = tau
        self.db = PersistenceDiagramDatabase(labels=[0, 1], dim=dim, tau=tau, maxdim=0, sample=None, alpha=alpha)
        self.distance = distance
    
    def fit(self, X, y, verbose=False):
        """ Fit the binary classification model. 
        Args:
            X: List[ndarray] (n_samples, ~n_signals)
            y: List[int] of labels (0 or 1)
        """
        for i, (xs, yi) in enumerate(zip(X, y)):
            self.db.add_signal(xs, label=yi)
            if verbose and (i % 10 == 0):
                print(f"Added signal {i}/{len(y)}", end='\r')
        print(f"Added signal {len(y)}/{len(y)}", end='\r')

    def predict_proba_sample(self, xs):
        """ Predict the probability for a single sample.
        Args:
            xs: ndarray (~n_signals,)
        Returns:
            prob: float in [0, 1]
        """
        xs_embeddings = takens_embedding(xs, dim=self.dim, tau=self.tau)
        xs_dgm = compute_persistence(xs_embeddings, maxdim=0)
        xs_dgm_h0 = xs_dgm[0]

        # Retrieve diagrams from the database
        E_h0 = self.db.get_diagrams(label=1, dim=0)
        N_h0 = self.db.get_diagrams(label=0, dim=0)

        # Compute distances
        dists_E = [self.distance(xs_dgm_h0, dgm) for dgm in E_h0]
        dists_N = [self.distance(xs_dgm_h0, dgm) for dgm in N_h0]

        mean_dist_E = np.mean(dists_E) if dists_E else np.inf
        mean_dist_N = np.mean(dists_N) if dists_N else np.inf

        #prob = np.exp(mean_dist_E) / (np.exp(mean_dist_E) + np.exp(mean_dist_N))
        prob = 1 / (1 + np.exp(mean_dist_N - mean_dist_E))
        return prob
    
    def predict_proba(self, X):
        """ Predict probabilities for multiple samples.
        Args:
            X: List[ndarray] (n_samples, ~n_signals)
        Returns:
            probs: List[float] of probabilities in [0, 1]
        """
        probs = [self.predict_proba_sample(xs) for xs in X]
        return probs

    def predict(self, X):
        """ Predict class labels for multiple samples.
        Args:
            X: List[ndarray] (n_samples, ~n_signals)
        Returns:
            labels: List[int] of predicted labels (0 or 1)
        """
        probs = self.predict_proba(X)
        labels = [1 if p >= 0.5 else 0 for p in probs]
        return labels

    def score(self, X, y):
        """ Compute the ROC AUC score.
        Args:
            X: List[ndarray] (n_samples, ~n_signals)
            y: List[int] of true labels (0 or 1)
        Returns:
            auc: float ROC AUC score
        """
        probs = self.predict_proba(X)
        auc = roc_auc_score(y, probs)
        return auc
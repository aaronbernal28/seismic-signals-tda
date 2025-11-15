""" Binary classification model implementation using MFCC features instead of Takens embeddings. """

import numpy as np
from src.databases import PersistenceDiagramDatabaseMFCC
from persim import bottleneck
from sklearn.metrics import roc_auc_score

class BinaryClassificationMFCC(PersistenceDiagramDatabaseMFCC):
    def __init__(self, distance=bottleneck, weigths=[1], n_mfcc=40, sr=40.0, win_length_sec=0.3, 
                 hop_length_sec=0.2, maxdim=None, sample=10, thresh=np.inf, alpha=0.1, max_points=500):
        """ Initialize the binary classification model using MFCC features.
        Args:
            distance: function to compute distance between persistence diagrams
            weigths: List[float] weights for each homology dimension
            n_mfcc: int, number of MFCC coefficients
            sr: float, sample rate of signals
            win_length_sec: float, window length in seconds for MFCC
            hop_length_sec: float, hop length in seconds for MFCC
            maxdim: int, maximum homology dimension to compute
            sample: int or None, number of diagrams to sample for distance computation
            thresh: float, threshold for diagram points
            alpha: float, proportion of points to keep after FPS subsampling
        """
        # If maxdim is provided, ensure it matches len(weigths)-1
        if maxdim is None:
            maxdim = len(weigths) - 1
        assert len(weigths) == maxdim + 1, "Weights length must match maxdim + 1"

        super().__init__(
            n_mfcc=n_mfcc,
            sr=sr,
            win_length_sec=win_length_sec,
            hop_length_sec=hop_length_sec,
            maxdim=maxdim,
            sample=sample,
            thresh=thresh,
            alpha=alpha,
            max_points=max_points
        )
        self.distance = distance
        self.weigths = np.array(weigths) / np.sum(weigths)  # Normalize weights
        # Keep a copy of init params for sklearn compatibility
        self._init_params = dict(distance=distance, weigths=list(weigths), n_mfcc=n_mfcc, sr=sr,
                                 win_length_sec=win_length_sec, hop_length_sec=hop_length_sec,
                                 maxdim=maxdim, sample=sample, thresh=thresh, alpha=alpha)
    
    def fit(self, X, y, verbose=False):
        """ Fit the binary classification model. 
        Args:
            X: List[ndarray] - each element is a 1D signal
            y: List[int] of labels (0 or 1)
        """
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

    # --- Scikit-learn compatibility ---
    def get_params(self, deep=True):
        params = dict(self._init_params)
        params.update(dict(
            distance=self.distance,
            weigths=list(self.weigths),
            n_mfcc=self.n_mfcc,
            sr=self.sr,
            win_length_sec=self.win_length / self.sr,
            hop_length_sec=self.hop_length / self.sr,
            maxdim=self.maxdim,
            sample=self.sample,
            thresh=self.thresh,
            alpha=self.alpha,
        ))
        return params

    def set_params(self, **params):
        for k, v in params.items():
            if k in self._init_params:
                self._init_params[k] = v

        distance = params.get('distance', self.distance)
        weigths = params.get('weigths', list(self.weigths))
        n_mfcc = params.get('n_mfcc', self.n_mfcc)
        sr = params.get('sr', self.sr)
        win_length_sec = params.get('win_length_sec', self.win_length / self.sr)
        hop_length_sec = params.get('hop_length_sec', self.hop_length / self.sr)
        maxdim = params.get('maxdim', self.maxdim)
        sample = params.get('sample', self.sample)
        thresh = params.get('thresh', self.thresh)
        alpha = params.get('alpha', self.alpha)

        if len(weigths) != maxdim + 1:
            raise ValueError("Weights length must match maxdim + 1")

        super().__init__(n_mfcc=n_mfcc, sr=sr, win_length_sec=win_length_sec, hop_length_sec=hop_length_sec,
                         maxdim=maxdim, sample=sample, thresh=thresh, alpha=alpha)
        self.distance = distance
        self.weigths = np.array(weigths) / np.sum(weigths)
        return self

    def predict_proba_sample(self, signal):
        """ Predict the probability for a single sample.
        Args:
            signal: ndarray - 1D signal array
        Returns:
            prob: float in [0, 1]
        """
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
        mean_dist_E = np.sum([w * d for w, d in zip(self.weigths, mean_dist_E)])
        mean_dist_N = np.sum([w * d for w, d in zip(self.weigths, mean_dist_N)])

        # Handle edge cases
        if mean_dist_E == np.inf and mean_dist_N == np.inf:
            prob = 0.5
        elif mean_dist_E == np.inf and mean_dist_N < np.inf:
            prob = 0.0
        elif mean_dist_E < np.inf and mean_dist_N == np.inf:
            prob = 1.0
        else:
            prob = 1 / (1 + np.exp(mean_dist_N - mean_dist_E))
        return prob
    
    def predict_proba(self, X):
        """ Predict probabilities for multiple samples.
        Args:
            X: List[ndarray] - each element is a 1D signal
        Returns:
            probs: List[float] of probabilities in [0, 1]
        """
        probs = [self.predict_proba_sample(signal) for signal in X]
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
        auc = roc_auc_score(y, probs)
        return auc
    
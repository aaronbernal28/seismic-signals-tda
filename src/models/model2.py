""" Binary classification model implementation. """

import numpy as np
from src.databases import PersistenceDiagramDatabaseTE
from persim import bottleneck
from sklearn.metrics import roc_auc_score

class BinaryClassificationTE(PersistenceDiagramDatabaseTE):
    def __init__(self, distance=bottleneck, weights=[1], dim=100, tau=10, stride=1, maxdim=None, sample=10, thresh=np.inf, alpha=0.1, max_points=500):
        """ Initialize the binary classification model.
        Args:
            distance: function to compute distance between persistence diagrams
            weights: List[float] weights for each homology dimension
            dim: int, embedding dimension
            tau: int, time delay
            stride: int, stride for sliding window
            sample: int or None, number of diagrams to sample for distance computation
            thresh: float, threshold for diagram points
            alpha: float, weight for birth-death transformation
        """
        # If maxdim is provided, ensure it matches len(weights)-1
        if maxdim is None:
            maxdim = len(weights) - 1
        assert len(weights) == maxdim + 1, "Weights length must match maxdim + 1"

        super().__init__(
            dim=dim, tau=tau,
            stride=stride,
            maxdim=maxdim,
            sample=sample,
            thresh=thresh,
            alpha=alpha,
            max_points=max_points)
        self.distance = distance
        self.weights = weights  # Store original weights for sklearn compatibility
        self._normalized_weights = np.array(weights)/np.sum(weights)  # Normalized weights for computation
        # Keep a copy of init params for sklearn compatibility
        self._init_params = dict(distance=distance, weights=list(weights), dim=dim, tau=tau,
                                 stride=stride, maxdim=maxdim, sample=sample, thresh=thresh, alpha=alpha)
    
    def fit(self, X, y, verbose=False):
        """ Fit the binary classification model. 
        Args:
            X: List[ndarray] (n_samples, ~n_signals)
            y: List[int] of labels (0 or 1)
        """
        for i, (xs, yi) in enumerate(zip(X, y)):
            self.add_signal(xs, label=yi)
            if verbose and (i % 10 == 0):
                print(f"Added signal {i}/{len(y)}", end='\r')
        
        if verbose:
            for label in [0, 1]:
                for d in range(self.maxdim + 1):
                    pc_len = []
                    for diagrams in self.get_diagrams(label=label, dim=d):
                        pc_len.append(len(diagrams))
                    # Print statistics
                    if pc_len:
                        print(f"Label {label}, Dim {d}: Mean diagram size: {np.mean(pc_len):.2f}, Std: {np.std(pc_len):.2f}, Max: {np.max(pc_len)}, Min: {np.min(pc_len)}")
        print("Model fitting complete." )

    # --- Scikit-learn compatibility ---
    def get_params(self, deep=True):
        params = dict(self._init_params)
        # Reflect any runtime changes
        params.update(dict(
            distance=self.distance,
            weights=list(self.weights),
            dim=self.dim,
            tau=self.tau,
            stride=self.stride,
            maxdim=self.maxdim,
            sample=self.sample,
            thresh=self.thresh,
            alpha=self.alpha,
        ))
        return params

    def set_params(self, **params):
        # Update known params
        for k, v in params.items():
            if k in self._init_params:
                self._init_params[k] = v
        # Rebuild internal state if structural params changed
        distance = params.get('distance', self.distance)
        weights = params.get('weights', list(self.weights))
        dim = params.get('dim', self.dim)
        tau = params.get('tau', self.tau)
        stride = params.get('stride', self.stride)
        maxdim = params.get('maxdim', self.maxdim)
        sample = params.get('sample', self.sample)
        thresh = params.get('thresh', self.thresh)
        alpha = params.get('alpha', self.alpha)

        # Validate weights vs maxdim
        if len(weights) != maxdim + 1:
            raise ValueError("Weights length must match maxdim + 1")

        # Reset base class state
        super().__init__(dim=dim, tau=tau, stride=stride, maxdim=maxdim, sample=sample, thresh=thresh, alpha=alpha)
        self.distance = distance
        self.weights = weights  # Store original weights
        self._normalized_weights = np.array(weights) / np.sum(weights)  # Normalized weights
        return self

    def predict_proba_sample(self, xs):
        """ Predict the probability for a single sample.
        Args:
            xs: ndarray (~n_signals,)
        Returns:
            prob: float in [0, 1]
        """
        # Compute persistence diagram for the input signal
        try:
            xs_dgm = self.transform(xs)
        except Exception as e:
            print(f"Error transforming the input signal: {e}")
            return 0.5  # Return a neutral probability in case of error

        mean_dist_E = []
        mean_dist_N = []

        for d in range(self.maxdim + 1):
            # Only proceed if there are points in the diagram
            if len(xs_dgm[d]) > 0:
                # Retrieve diagrams from the database
                E_h0 = self.get_diagrams(label=1, dim=d)
                N_h0 = self.get_diagrams(label=0, dim=d)
                #print(f"Dimension {d}: {len(E_h0)} diagrams for class 1, {len(N_h0)} diagrams for class 0.", end='\r')
                # Compute distances
                dists_E = [self.distance(xs_dgm[d], dgm) for dgm in E_h0]
                dists_N = [self.distance(xs_dgm[d], dgm) for dgm in N_h0]

                mean_dist_E.append(np.mean(dists_E) if dists_E else np.inf) 
                mean_dist_N.append(np.mean(dists_N) if dists_N else np.inf)
            else:
                mean_dist_E.append(np.inf) 
                mean_dist_N.append(np.inf) 

        #prob = np.exp(mean_dist_E) / (np.exp(mean_dist_E) + np.exp(mean_dist_N))
        # Ponderate distances with weights between homology dimensions
        mean_dist_E = np.sum([w * d for w, d in zip(self._normalized_weights, mean_dist_E)])
        mean_dist_N = np.sum([w * d for w, d in zip(self._normalized_weights, mean_dist_N)])

        # Just to be sure
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
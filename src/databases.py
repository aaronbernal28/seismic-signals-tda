import numpy as np
from src.utils import takens_embedding, compute_persistence
from gudhi.subsampling import choose_n_farthest_points as fps
from gtda.time_series import TakensEmbedding

class PersistenceDiagramDatabase:
    def __init__(self, labels=[0, 1], dim=100, tau=10, stride=1, maxdim=2, sample=None, thresh=np.inf, alpha=0.1):
        """Initialize the persistence diagram database.
        Args:
            labels (list): List of unique identifiers for diagrams.
            dim (int): Embedding dimension for Takens' embedding.
            tau (int): Time delay for Takens' embedding.
            stride (int): Stride for Takens' embedding.
            maxdim (int): Maximum homology dimension to compute.
            sample (int, optional): If provided, randomly sample this many diagrams when retrieving.
            thresh (float): Threshold for Vietoris-Rips complex.
            alpha (float): Proportion of subsampling.
        """
        self.labels = labels
        self.maxdim = maxdim
        self.db = {label: {d: [] for d in range(maxdim + 1)} for label in labels}
        self.seed = 28
        self.sample = sample
        self.thresh = thresh
        self.alpha = alpha
        self.TE = TakensEmbedding(time_delay=tau, dimension=dim, stride=stride)

    def add_signal(self, signal, label):
        """Compute and add the persistence diagram of a signal to the database.
        
        Args:
            signal (np.ndarray): 1D array of the time series signal.
            label (int): Unique identifier for the diagram.
        """
        diagrams = self.transform(signal)
        
        try:
            # Add each persistence diagram to the database.
            for dim in range(len(diagrams)):
                self.db[label][dim].append(diagrams[dim])
        except TypeError:
            print(f"Error adding signal with label {label}: diagrams is None")
    
    def get_diagrams(self, label, dim=0):
        """Retrieve all persistence diagrams for a given label and dimension.
        
        Args:
            label (int): Unique identifier for the diagram.
            dim (int): Homology dimension.
            sample (int, optional): If provided, randomly sample this many diagrams.
        Returns:
            list: List of persistence diagrams, each as np.ndarray of shape (~N, 2)
        """
        output = self.db[label][dim]  # Keep as list since diagrams have varying shapes
        if self.sample is not None:
            indices = np.random.choice(len(output), size=min(self.sample, len(output)), replace=False)
            output = [output[i] for i in indices]
        return output
    
    def transform(self, signal):
        """Compute the persistence diagrams for a given signal. 
        Args:
            signal (np.ndarray): 1D array of the time series signal.
        Returns:
            list: List of persistence diagrams, each as np.ndarray of shape (~N, 2)
        """
        # Compute Takens' embedding
        try:
            # TakensEmbedding expects 2D input: (n_samples, n_timestamps)
            # Reshape 1D signal to 2D if necessary
            if signal.ndim == 1:
                signal_2d = signal.reshape(1, -1)
            else:
                signal_2d = signal
            
            # Use fit_transform instead of transform (no need to call fit separately)
            embedded = self.TE.fit_transform(signal_2d)
            
            # fit_transform returns shape (n_samples, n_windows, n_dims)
            # For a single signal, squeeze to get (n_windows, n_dims)
            if embedded.shape[0] == 1:
                embedded = embedded[0]
        except ValueError as e:
            print(f"Error with Takens embeddings: {e}")
            return None
        #print("Embedded shape:", embedded.shape, end="\n")

        # Subsample the embedded points using farthest point sampling
        # fps returns the actual subsampled points, not indices
        nb_points = max(1, int(self.alpha * embedded.shape[0]))  # Ensure at least 1 point
        embedded_sparse = np.array(fps(embedded, nb_points=nb_points, starting_point=None))
        
        # Compute persistence diagrams
        diagrams = compute_persistence(embedded_sparse, maxdim=self.maxdim, thresh=self.thresh)
        return diagrams
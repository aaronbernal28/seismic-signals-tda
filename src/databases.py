import numpy as np
from src.utils import takens_embedding, compute_persistence
from gudhi.subsampling import choose_n_farthest_points as fps

class PersistenceDiagramDatabase:
    def __init__(self, labels=[0, 1], dim=100, tau=10, maxdim=2, sample=None, thresh=np.inf, alpha=0.1):
        """Initialize the persistence diagram database.
        Args:
            labels (list): List of unique identifiers for diagrams.
            dim (int): Embedding dimension for Takens' embedding.
            tau (int): Time delay for Takens' embedding.
            maxdim (int): Maximum homology dimension to compute.
            sample (int, optional): If provided, randomly sample this many diagrams when retrieving.
            thresh (float): Threshold for Vietoris-Rips complex.
            alpha (float): Proportion of subsampling.
        """
        self.labels = labels
        self.dim = dim
        self.tau = tau
        self.maxdim = maxdim
        self.db = {label: {dim: [] for dim in range(maxdim + 1)} for label in labels}
        self.seed = 28
        self.sample = sample
        self.thresh = thresh
        self.alpha = alpha

    def add_diagram(self, diagram, label, dim=0):
        """Add a persistence diagram to the database.
        
        Args:
            label (int): Unique identifier for the diagram.
                1 : Earthquake
                0 : Noise
            diagram (np.ndarray): Persistence diagram to store (N x 2 array).
        """
        self.db[label][dim].append(diagram)

    def add_signal(self, signal, label):
        """Compute and add the persistence diagram of a signal to the database.
        
        Args:
            signal (np.ndarray): 1D array of the time series signal.
            label (int): Unique identifier for the diagram.
        """
        # Compute Takens' embedding
        embedded = takens_embedding(signal, dim=self.dim, tau=self.tau)
        #print("Embedded shape:", embedded.shape, end="\n")

        # Subsample the embedded points using farthest point sampling
        embedded = np.array(fps(embedded, nb_points=int(self.alpha * embedded.shape[0]), starting_point=None))
        
        # Compute persistence diagrams
        diagrams = compute_persistence(embedded, maxdim=self.maxdim, thresh=self.thresh)
        
        # Store diagrams in the database
        for dim in range(len(diagrams)):
            self.add_diagram(diagrams[dim], label, dim)
    
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
            np.random.seed(self.seed)
            indices = np.random.choice(len(output), size=min(self.sample, len(output)), replace=False)
            output = [output[i] for i in indices]
            self.seed += 1
        return output
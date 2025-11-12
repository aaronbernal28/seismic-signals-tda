import numpy as np
from src.utils import takens_embedding, compute_persistence

class PersistenceDiagramDatabase:
    def __init__(self, labels=[0, 1], dim=4, tau=10, maxdim=2, sample=None):
        """Initialize the persistence diagram database.
        Args:
            labels (list): List of unique identifiers for diagrams.
            dim (int): Embedding dimension for Takens' embedding.
            tau (int): Time delay for Takens' embedding.
            maxdim (int): Maximum homology dimension to compute.
            sample (int, optional): If provided, randomly sample this many diagrams when retrieving.
        """
        self.labels = labels
        self.dim = dim
        self.tau = tau
        self.maxdim = maxdim
        self.db = {label: {dim: [] for dim in range(maxdim + 1)} for label in labels}
        self.seed = 28
        self.sample = sample

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
        embedded = takens_embedding(signal, dim=self.dim, delay=self.tau)
        diagrams = compute_persistence(embedded, maxdim=self.maxdim)
        for dim in range(len(diagrams)):
            self.add_diagram(diagrams[dim], label, dim)
    
    def get_diagrams(self, label, dim=0):
        """Retrieve all persistence diagrams for a given label and dimension.
        
        Args:
            label (int): Unique identifier for the diagram.
            dim (int): Homology dimension.
            sample (int, optional): If provided, randomly sample this many diagrams.
        Returns:
            np.ndarray: Array of persistence diagrams. (M x ~N x 2)
        """
        output = np.array(self.db[label][dim])
        if self.sample is not None:
            np.random.seed(self.seed)
            output = np.random.choice(output, size=self.sample, replace=False, axis=0)
        return output
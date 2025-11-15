import numpy as np
from src.utils import compute_persistence
from gudhi.subsampling import choose_n_farthest_points as fps
from gtda.time_series import TakensEmbedding
import librosa

class PersistenceDiagramDatabaseTE:
    def __init__(self, labels=[0, 1], dim=100, tau=10, stride=1, maxdim=2, sample=None, thresh=np.inf, alpha=0.1, max_points=500):
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
        self.max_points = max_points
        # Store TE parameters explicitly for downstream access (e.g., grid search)
        self.dim = dim
        self.tau = tau
        self.stride = stride
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
        nb_points = max(1, min(int(self.alpha * embedded.shape[0]), self.max_points))  # Ensure at least 1 point
        embedded_sparse = np.array(fps(embedded, nb_points=nb_points, starting_point=None))
        
        # Compute persistence diagrams
        diagrams = compute_persistence(embedded_sparse, maxdim=self.maxdim, thresh=self.thresh)
        return diagrams
    

class PersistenceDiagramDatabaseMFCC:
    def __init__(self, labels=[0, 1], n_mfcc=40, sr=40.0, win_length_sec=0.3, hop_length_sec=0.2, 
                 maxdim=2, sample=None, thresh=np.inf, alpha=0.1, max_points=500):
        """Initialize the persistence diagram database for MFCC features.
        Args:
            labels (list): List of unique identifiers for diagrams.
            n_mfcc (int): Number of MFCC coefficients to compute.
            sr (float): Sample rate of the signal.
            win_length_sec (float): Window length in seconds for MFCC.
            hop_length_sec (float): Hop length in seconds for MFCC.
            maxdim (int): Maximum homology dimension to compute.
            sample (int, optional): If provided, randomly sample this many diagrams when retrieving.
            thresh (float): Threshold for Vietoris-Rips complex.
            alpha (float): Proportion of subsampling for FPS.
            max_points (int): Maximum number of points to keep after subsampling.
        """
        self.labels = labels
        self.maxdim = maxdim
        self.db = {label: {d: [] for d in range(maxdim + 1)} for label in labels}
        self.seed = 28
        self.sample = sample
        self.thresh = thresh
        self.alpha = alpha
        self.max_points = max_points
        
        # MFCC parameters
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.win_length = int(win_length_sec * sr)
        self.hop_length = max(1, int(hop_length_sec * sr))

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
        """Compute the persistence diagrams for a given signal using MFCC features.
        Args:
            signal (np.ndarray): 1D array of the time series signal.
        Returns:
            list: List of persistence diagrams, each as np.ndarray of shape (~N, 2)
        """
        
        try:
            # Convert signal to float32
            x = signal.astype(np.float32).copy()
            
            # Compute MFCC features
            mfcc = librosa.feature.mfcc(
                y=x,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.win_length,
                hop_length=self.hop_length,
                n_mels=128,
                dct_type=2,
                norm='ortho',
                center=False
            )
            
            # Compute delta and delta-delta features with adaptive width
            width = min(mfcc.shape[1], 3) if mfcc.shape[1] < 9 else 9
            if width % 2 == 0:
                width -= 1
            width = max(3, width)
            
            mfcc_delta = librosa.feature.delta(mfcc, width=width, mode='nearest', order=1)
            mfcc_delta_delta = librosa.feature.delta(mfcc, width=width, mode='nearest', order=2)
            
            # Concatenate MFCC, delta, and delta-delta
            mfcc_combined = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta), axis=0)
            
            # Normalize
            mfcc_combined_norm = (mfcc_combined - np.mean(mfcc_combined, axis=1, keepdims=True)) / \
                                 (np.std(mfcc_combined, axis=1, keepdims=True) + 1e-8)
            
            # Transpose to get (frames, emb_dim) shape
            point_cloud = mfcc_combined_norm.T
            
        except Exception as e:
            print(f"Error computing MFCC features: {e}")
            return None
        
        # Apply FPS subsampling
        nb_points = max(1, min(int(self.alpha * point_cloud.shape[0]), self.max_points))
        if nb_points < point_cloud.shape[0]:
            point_cloud_sparse = np.array(fps(point_cloud, nb_points=nb_points, starting_point=None))
        else:
            point_cloud_sparse = point_cloud
        
        # Compute persistence diagrams
        diagrams = compute_persistence(point_cloud_sparse, maxdim=self.maxdim, thresh=self.thresh)
        return diagrams
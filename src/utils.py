from pathlib import Path
from obspy.clients.fdsn import Client
from config.config import *
import src.ecg as ecg
import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_waveform(signal, label='Earthquake', mag=0):
    """Plot waveform for a given interval."""
    time = np.arange(len(signal))  # Assume time steps starting from 0
    plt.plot(time, signal, linewidth=0.5, color='darkblue')
    plt.title(f'{label} - Mag: {mag}', fontsize=11, fontweight='bold')
    plt.xlabel('Step (40Hz/s)', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(signal):.2e}\nStd: {np.std(signal):.2e}\nMin: {np.min(signal):.2e}\nMax: {np.max(signal):.2e}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

class SeismicDataset:
    """Dataset class for seismic signals."""
    def __init__(self, data_path=r"data\processed\signals_train.hdf5", seed=None):
        self.data_path = Path(data_path)
        self.signals = []
        self.labels = []
        self.mags = []
        # Per-instance RNG; no global reseeding
        self.rng = np.random.default_rng(seed)
        self._load_data()

    def _load_data(self):
        """Load seismic signals and labels from the data path."""
        with h5py.File(self.data_path, 'r') as hf:
            # Load metadata to get mag values
            metadata_ids = hf['metadata']['id'][:]
            metadata_mags = hf['metadata']['mag'][:]
            
            # Decode bytes to strings if necessary
            if metadata_ids.dtype.kind == 'S':
                metadata_ids = [id.decode('utf-8') if id else '' for id in metadata_ids]
            else:
                metadata_ids = list(metadata_ids)
            
            # Create a mapping from interval_id to mag
            id_to_mag = dict(zip(metadata_ids, metadata_mags))
            
            # Load signals
            for interval_id in hf['signals'].keys():
                sig_group = hf['signals'][interval_id]
                label = sig_group.attrs.get('label', 0)
                
                # Get mag from metadata using interval_id
                mag = id_to_mag.get(interval_id, float('nan'))
                
                # Convert NaN to 0.0 for non-events
                if np.isnan(mag):
                    mag = 0.0

                # Get first trace data
                trace_name = list(sig_group.keys())[0]
                data = sig_group[trace_name]['data'][:]
                
                self.signals.append(data)
                self.labels.append(label)
                self.mags.append(mag)
        
        # Shuffle the data using per-instance RNG (optional reproducibility via seed)
        indices = np.arange(len(self.signals))
        self.rng.shuffle(indices)
        self.signals = [self.signals[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.mags = [self.mags[i] for i in indices]

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Support slicing like dataset[:10]
            return [self.signals[i] for i in range(*idx.indices(len(self)))], \
                   [self.labels[i] for i in range(*idx.indices(len(self)))], \
                   [self.mags[i] for i in range(*idx.indices(len(self)))]
        return self.signals[idx], self.labels[idx], self.mags[idx]
    
    def get_data(self):
        """Get all signals, labels, and mags as numpy arrays."""
        return self.signals, np.array(self.labels), np.array(self.mags)

def download_waveforms():
    """Download seismic waveform data."""
    print("Initializing FDSN client...")
    client = Client(DATA_CENTER)
    
    print(f"Fetching waveform data for station {STATION_CODE}...")
    try:
        st = client.get_waveforms(
            network=NETWORK,
            station=STATION_CODE,
            location="*",
            channel=CHANNEL,
            starttime=START_TIME,
            endtime=END_TIME
        )
        
        # Save to file
        output_file = Path(RAW_DATA_PATH) / "waveforms.mseed"
        st.write(str(output_file), format='MSEED')
        
        print(f"✓ Waveform data saved to {output_file}")
        print(f"  {len(st)} traces downloaded")
        return st
        
    except Exception as e:
        print(f"✗ Error downloading waveforms: {e}")
        return None


def takens_embedding(signal, dim, tau):
    """Takens' embedding
    Args:
        signal (np.ndarray): 1D array of the time series signal.
        dim (int): Embedding dimension.
        tau (int): Time delay.
    Returns:
        np.ndarray: 2D array of shape (m, dim) where m = n - (dim - 1) * tau.
    """
    return ecg.takens_embedding(signal, dim, tau)

def compute_persistence(point_cloud, maxdim=1, thresh=np.inf, metric='euclidean'):
    """Compute persistence diagrams using Ripser.
    
    Args:
        point_cloud (np.ndarray): 2D array of shape (n_points, n_dimensions) representing the point cloud.
        maxdim (int): Maximum homology dimension to compute. Default is 1.
        thresh (float): Maximum filtration value. Default is infinity.
    Returns:
        list: List of persistence diagrams, one for each dimension up to maxdim.
    """
    return ecg.compute_persistence(point_cloud, maxdim, thresh, metric)

def bottleneck_distance(dgm1, dgm2):
    """Compute the bottleneck distance between two persistence diagrams.
    
    Args:
        dgm1 (np.ndarray): First persistence diagram of shape (n_points, 2).
        dgm2 (np.ndarray): Second persistence diagram of shape (n_points, 2).
    
    Returns:
        float: Bottleneck distance between the two diagrams.
    """
    return ecg.bottleneck_distance(dgm1, dgm2)

def compute_distances(dgm1, dgm2):
    """Compute bottleneck and Wasserstein distances between two persistence diagrams.
    
    Args:
        dgm1 (np.ndarray): First persistence diagram of shape (n_points, 2).
        dgm2 (np.ndarray): Second persistence diagram of shape (n_points, 2).
    
    Returns:
        tuple: (bottleneck_distance, wasserstein_distance). Returns (inf, inf) if either diagram is empty.
    """
    return ecg.compute_distances(dgm1, dgm2)

def load_datasets(train_path=None, test_path=None, seed=None):
    """Load training and test datasets.
    
    Args:
        train_path: Path to training dataset (default: from config)
        test_path: Path to test dataset (default: from config)
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    if train_path is None:
        train_path = TRAIN_DATA_PATH
    if test_path is None:
        test_path = TEST_DATA_PATH
    
    print("=" * 70)
    print("Loading Datasets")
    print("=" * 70)
    
    train_dataset = SeismicDataset(data_path=Path(train_path), seed=seed)
    test_dataset = SeismicDataset(data_path=Path(test_path), seed=seed)
    
    X_train, y_train, mag_train = train_dataset.get_data()
    X_test, y_test, mag_test = test_dataset.get_data()
    
    print(f"✓ Datasets loaded successfully")
    print(f"  Train: {len(train_dataset)} signals")
    print(f"  Test: {len(test_dataset)} signals")
    print(f"  Train labels: {np.bincount(y_train)}")
    print(f"  Test labels: {np.bincount(y_test)}")
    
    return X_train, y_train, X_test, y_test
from pathlib import Path
from obspy.clients.fdsn import Client
from config.config import *
import src.ecg as ecg
import h5py

class SeismicDataset:
    """Dataset class for seismic signals."""
    def __init__(self, data_path=r"data\processed\signals_train.hdf5"):
        self.data_path = Path(data_path)
        self.signals = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        """Load seismic signals and labels from the data path."""
        with h5py.File(self.data_path, 'r') as hf:
            for interval_id in hf['signals'].keys():
                sig_group = hf['signals'][interval_id]
                label = sig_group.attrs.get('label', 0)
                
                # Get first trace data
                trace_name = list(sig_group.keys())[0]
                data = sig_group[trace_name]['data'][:]
                
                self.signals.append(data)
                self.labels.append(label)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Support slicing like dataset[:10]
            return [(self.signals[i], self.labels[i]) for i in range(*idx.indices(len(self)))]
        return self.signals[idx], self.labels[idx]

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


def takens_embedding(signal, d, tau):
    """Takens' embedding
    Args:
        signal (np.ndarray): 1D array of the time series signal.
        d (int): Embedding dimension.
        tau (int): Time delay.
    Returns:
        np.ndarray: 2D array of shape (m, d) where m = n - (d - 1) * tau.
    """
    return ecg.takens_embedding(signal, d, tau)

def compute_persistence(point_cloud, maxdim=1):
    """Compute persistence diagrams using Ripser.
    
    Args:
        point_cloud (np.ndarray): 2D array of shape (n_points, n_dimensions) representing the point cloud.
        maxdim (int): Maximum homology dimension to compute. Default is 1.
    
    Returns:
        list: List of persistence diagrams, one for each dimension up to maxdim.
    """
    return ecg.compute_persistence(point_cloud, maxdim)

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
from pathlib import Path
from obspy.clients.fdsn import Client
from config.config import *
import src.ecg as ecg
import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_waveform(signal, label='Earthquake', mag=0):
    """Graficar forma de onda para un intervalo dado."""
    time = np.arange(len(signal))  # Asumir pasos de tiempo empezando desde 0
    plt.plot(time, signal, linewidth=0.5, color='darkblue')
    plt.title(f'{label} - Mag: {mag}', fontsize=11, fontweight='bold')
    plt.xlabel('Paso (40Hz/s)', fontsize=10)
    plt.ylabel('Amplitud', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Agregar texto de estadísticas
    stats_text = f'Media: {np.mean(signal):.2e}\nDesv: {np.std(signal):.2e}\nMín: {np.min(signal):.2e}\nMáx: {np.max(signal):.2e}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

class SeismicDataset:
    """Clase de dataset para señales sísmicas."""
    def __init__(self, data_path=r"data\processed\signals_train.hdf5", seed=None):
        self.data_path = Path(data_path)
        self.signals = []
        self.labels = []
        self.mags = []
        # RNG por instancia; sin reseeding global
        self.rng = np.random.default_rng(seed)
        self._load_data()

    def _load_data(self):
        """Cargar señales sísmicas y etiquetas desde la ruta de datos."""
        with h5py.File(self.data_path, 'r') as hf:
            # Cargar metadata para obtener valores de mag
            metadata_ids = hf['metadata']['id'][:]
            metadata_mags = hf['metadata']['mag'][:]
            
            # Decodificar bytes a strings si es necesario
            if metadata_ids.dtype.kind == 'S':
                metadata_ids = [id.decode('utf-8') if id else '' for id in metadata_ids]
            else:
                metadata_ids = list(metadata_ids)
            
            # Crear mapeo de interval_id a mag
            id_to_mag = dict(zip(metadata_ids, metadata_mags))
            
            # Cargar señales
            for interval_id in hf['signals'].keys():
                sig_group = hf['signals'][interval_id]
                label = sig_group.attrs.get('label', 0)
                
                # Obtener mag desde metadata usando interval_id
                mag = id_to_mag.get(interval_id, float('nan'))
                
                # Convertir NaN a 0.0 para no-eventos
                if np.isnan(mag):
                    mag = 0.0

                # Iterar sobre TODAS las trazas (ej. 'trace_0', 'trace_1')
                # para este intervalo y agregar cada una como señal.
                for trace_name in sig_group.keys():
                    # Verificar si la clave es un grupo y contiene 'data'
                    if isinstance(sig_group[trace_name], h5py.Group) and 'data' in sig_group[trace_name]:
                        data = sig_group[trace_name]['data'][:]
                        
                        self.signals.append(data)
                        self.labels.append(label)
                        self.mags.append(mag)
        
        # Mezclar los datos usando RNG por instancia (reproducibilidad opcional via seed)
        indices = np.arange(len(self.signals))
        self.rng.shuffle(indices)
        self.signals = [self.signals[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.mags = [self.mags[i] for i in indices]

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Soporte para slicing como dataset[:10]
            return [self.signals[i] for i in range(*idx.indices(len(self)))], \
                   [self.labels[i] for i in range(*idx.indices(len(self)))], \
                   [self.mags[i] for i in range(*idx.indices(len(self)))]
        return self.signals[idx], self.labels[idx], self.mags[idx]
    
    def get_data(self, max_samples=None):
        """Obtener todas las señales, etiquetas y mags como arreglos numpy."""
        if max_samples is not None:
            return self.signals[:max_samples], np.array(self.labels[:max_samples]), np.array(self.mags[:max_samples])
        return self.signals, np.array(self.labels), np.array(self.mags)

def download_waveforms():
    """Descargar datos de formas de onda sísmicas."""
    print("Inicializando cliente FDSN...")
    client = Client(DATA_CENTER)
    
    print(f"Obteniendo datos de forma de onda para la estación {STATION_CODE}...")
    try:
        st = client.get_waveforms(
            network=NETWORK,
            station=STATION_CODE,
            location="*",
            channel=CHANNEL,
            starttime=START_TIME,
            endtime=END_TIME
        )
        
        # Guardar en archivo
        output_file = Path(RAW_DATA_PATH) / "waveforms.mseed"
        st.write(str(output_file), format='MSEED')
        
        print(f"✓ Datos de forma de onda guardados en {output_file}")
        print(f"  {len(st)} trazas descargadas")
        return st
        
    except Exception as e:
        print(f"✗ Error descargando formas de onda: {e}")
        return None


def takens_embedding(signal, dim, tau):
    """Incrustación de Takens
    Args:
        signal (np.ndarray): Arreglo 1D de la serie temporal.
        dim (int): Dimensión de incrustación.
        tau (int): Retardo temporal.
    Returns:
        np.ndarray: Arreglo 2D de forma (m, dim) donde m = n - (dim - 1) * tau.
    """
    return ecg.takens_embedding(signal, dim, tau)

def compute_persistence(point_cloud, maxdim=1, thresh=np.inf, metric='euclidean'):
    """Calcular diagramas de persistencia usando Ripser.
    
    Args:
        point_cloud (np.ndarray): Arreglo 2D de forma (n_points, n_dimensions) representando la nube de puntos.
        maxdim (int): Dimensión máxima de homología a calcular. Por defecto es 1.
        thresh (float): Valor máximo de filtración. Por defecto es infinito.
    Returns:
        list: Lista de diagramas de persistencia, uno por cada dimensión hasta maxdim.
    """
    return ecg.compute_persistence(point_cloud, maxdim, thresh, metric)

def bottleneck_distance(dgm1, dgm2):
    """Calcular la distancia bottleneck entre dos diagramas de persistencia.
    
    Args:
        dgm1 (np.ndarray): Primer diagrama de persistencia de forma (n_points, 2).
        dgm2 (np.ndarray): Segundo diagrama de persistencia de forma (n_points, 2).
    
    Returns:
        float: Distancia bottleneck entre los dos diagramas.
    """
    return ecg.bottleneck_distance(dgm1, dgm2)

def compute_distances(dgm1, dgm2):
    """Calcular distancias bottleneck y Wasserstein entre dos diagramas de persistencia.
    
    Args:
        dgm1 (np.ndarray): Primer diagrama de persistencia de forma (n_points, 2).
        dgm2 (np.ndarray): Segundo diagrama de persistencia de forma (n_points, 2).
    
    Returns:
        tuple: (bottleneck_distance, wasserstein_distance). Retorna (inf, inf) si algún diagrama está vacío.
    """
    return ecg.compute_distances(dgm1, dgm2)

def validate_signal_length(signal, dim, tau):
    """Validar que una señal tenga suficientes timestamps para Takens embedding.
    
    Args:
        signal (np.ndarray): Señal de entrada.
        dim (int): Dimensión de incrustación.
        tau (int): Retardo temporal.
    
    Returns:
        bool: True si la señal es válida, False si es muy corta.
    """
    # Mínimo necesario: dim + (dim-1)*tau timestamps
    min_length = dim + (dim - 1) * tau
    return len(signal) >= min_length

def filter_valid_signals(X, y, dim, tau, verbose=True):
    """Filtrar señales que son muy cortas para Takens embedding.
    
    Args:
        X (list): Lista de señales.
        y (np.ndarray): Etiquetas.
        dim (int): Dimensión de incrustación.
        tau (int): Retardo temporal.
        verbose (bool): Imprimir información de filtrado.
    
    Returns:
        tuple: (X_filtered, y_filtered, num_removed)
    """
    min_length = dim + (dim - 1) * tau
    valid_indices = [i for i, signal in enumerate(X) if validate_signal_length(signal, dim, tau)]
    
    num_removed = len(X) - len(valid_indices)
    
    if num_removed > 0 and verbose:
        removed_by_label = {}
        for i, (signal, label) in enumerate(zip(X, y)):
            if i not in valid_indices:
                removed_by_label[label] = removed_by_label.get(label, 0) + 1
        
        print(f"\n⚠ ADVERTENCIA: {num_removed} señal(es) removida(s) (longitud < {min_length})")
        for label, count in removed_by_label.items():
            label_name = 'Terremoto' if label == 1 else 'Ruido'
            print(f"  - {label_name} (etiqueta {label}): {count} señal(es)")
    
    X_filtered = [X[i] for i in valid_indices]
    y_filtered = y[valid_indices]
    
    return X_filtered, y_filtered, num_removed

def load_datasets(train_path=None, test_path=None, seed=None, max_samples=None):
    """Cargar datasets de entrenamiento y prueba.
    
    Args:
        train_path: Ruta al dataset de entrenamiento (por defecto: desde config)
        test_path: Ruta al dataset de prueba (por defecto: desde config)
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    if train_path is None:
        train_path = TRAIN_DATA_PATH
    if test_path is None:
        test_path = TEST_DATA_PATH
    
    print("=" * 70)
    print("Cargando Datasets")
    print("=" * 70)
    
    train_dataset = SeismicDataset(data_path=Path(train_path), seed=seed)
    test_dataset = SeismicDataset(data_path=Path(test_path), seed=seed)
    
    X_train, y_train, mag_train = train_dataset.get_data(max_samples=max_samples)
    X_test, y_test, mag_test = test_dataset.get_data(max_samples=int((max_samples)*0.2) if max_samples is not None else None)
    
    print(f"✓ Datasets cargados exitosamente")
    print(f"  Entrenamiento: {len(train_dataset)} señales")
    print(f"  Prueba: {len(test_dataset)} señales")
    print(f"  Etiquetas entrenamiento: {np.bincount(y_train)}")
    print(f"  Etiquetas prueba: {np.bincount(y_test)}")
    
    return X_train, y_train, X_test, y_test
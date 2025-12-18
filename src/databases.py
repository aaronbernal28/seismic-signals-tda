import numpy as np
from src.utils import compute_persistence, normalize_minmax
from gudhi.subsampling import choose_n_farthest_points as fps
from gtda.time_series import TakensEmbedding
import librosa
from src.cache import get_cache

class PersistenceDiagramDatabaseTE:
    def __init__(self, labels=[0, 1], dim=100, tau=10, stride=1, maxdim=2, sample=None, thresh=np.inf, alpha=0.1, max_points=500, seed=28, normalize_minmax=True):
        """Inicializar la base de datos de diagramas de persistencia.
        Args:
            labels (list): Lista de identificadores únicos para los diagramas.
            dim (int): Dimensión de la incrustación de Takens.
            tau (int): Retardo temporal para la incrustación de Takens.
            stride (int): Paso para la incrustación de Takens.
            maxdim (int): Dimensión máxima de homología a calcular.
            sample (int, opcional): Si se indica, muestrea aleatoriamente esa cantidad de diagramas.
            thresh (float): Umbral para el complejo de Vietoris-Rips.
            alpha (float): Proporción de subsampling.
            normalize_minmax (bool): Si True, normaliza cada señal a [0, 1] antes de Takens.
        """
        self.labels = labels
        self.maxdim = maxdim
        self.db = {label: {d: [] for d in range(maxdim + 1)} for label in labels}
        self.seed = seed
        self.sample = sample
        self.thresh = thresh
        self.alpha = alpha
        self.max_points = max_points
        self.normalize_minmax = normalize_minmax
        # Guardar parámetros TE explícitamente para usos posteriores (por ejemplo, grid search)
        self.dim = dim
        self.tau = tau
        self.stride = stride
        self.TE = TakensEmbedding(time_delay=tau, dimension=dim, stride=stride)
        # Generador RNG para muestreo
        self.rng = np.random.default_rng(self.seed)

    def add_signal(self, signal, label):
        """Calcular y agregar el diagrama de persistencia de una señal en la base de datos.
        
        Args:
            signal (np.ndarray): Arreglo 1D de la serie temporal.
            label (int): Identificador único del diagrama.
        
        Returns:
            bool: True si se agregó exitosamente, False si falló.
        """
        diagrams = self.transform(signal)
        
        try:
            # Agregar cada diagrama de persistencia a la base de datos.
            for dim in range(len(diagrams)):
                self.db[label][dim].append(diagrams[dim])
            return True
        except TypeError:
            print(f"Error al agregar la señal con etiqueta {label}: diagrams es None")
            return False
    
    def get_diagrams(self, label, dim=0, predict_mode=False):
        """Recuperar todos los diagramas de persistencia para una etiqueta y dimensión dadas.
        
        Args:
            label (int): Identificador único del diagrama.
            dim (int): Dimensión de homología.
            predict_mode (bool): Si True, usar todos los diagramas. Si False, aplicar muestreo.
        Returns:
            list: Lista de diagramas de persistencia, cada uno como np.ndarray de forma (~N, 2)
        """
        output = self.db[label][dim]  # Mantener como lista ya que los diagramas tienen formas variables
        # Durante predicción, usar TODOS los diagramas para reducir regularidad y mejorar estimaciones
        if not predict_mode and self.sample is not None and len(output) > 0:
            indices = self.rng.choice(len(output), size=min(self.sample, len(output)), replace=False)
            output = [output[i] for i in indices]
        return output
    
    def transform(self, signal):
        """Calcular los diagramas de persistencia para una señal dada.
        Args:
            signal (np.ndarray): Arreglo 1D de la serie temporal.
        Returns:
            list: Lista de diagramas de persistencia, cada uno como np.ndarray de forma (~N, 2)
        """
        # Verificar el caché primero
        cache = get_cache()
        cached = cache.get_te(
            signal, self.dim, self.tau, self.stride,
            self.maxdim, self.thresh, self.alpha, self.max_points
        )
        if cached is not None:
            return cached
        
        # Calcular la incrustación de Takens
        try:
            x = normalize_minmax(signal) if self.normalize_minmax else signal.astype(np.float32).copy()
            # TakensEmbedding requiere entrada 2D: (n_samples, n_timestamps)
            # Reajustar señal 1D a 2D si hace falta
            if x.ndim == 1:
                signal_2d = x.reshape(1, -1)
            else:
                signal_2d = x
            
            # Usar fit_transform en lugar de transform (no es necesario llamar a fit por separado)
            embedded = self.TE.fit_transform(signal_2d)
            
            # fit_transform devuelve forma (n_samples, n_windows, n_dims)
            # Para una señal única, se aplana para obtener (n_windows, n_dims)
            if embedded.shape[0] == 1:
                embedded = embedded[0]
        except ValueError as e:
            print(f"Error con las incrustaciones de Takens: {e}")
            return None
        #print("Embedded shape:", embedded.shape, end="\n")

        # Submuestrear puntos incrustados con FPS (farthest point sampling)
        # fps devuelve los puntos subsampleados reales, no índices
        nb_points = max(1, min(int(self.alpha * embedded.shape[0]), self.max_points))  # Asegurar al menos 1 punto
        embedded_sparse = np.array(fps(embedded, nb_points=nb_points, starting_point=None))
        
        # Calcular diagramas de persistencia
        diagrams = compute_persistence(embedded_sparse, maxdim=self.maxdim, thresh=self.thresh)
        
        # Guardar en el caché
        cache.put_te(
            signal, self.dim, self.tau, self.stride,
            self.maxdim, self.thresh, self.alpha, self.max_points,
            diagrams
        )
        
        return diagrams
    

class PersistenceDiagramDatabaseMFCC:
    def __init__(self, labels=[0, 1], n_mfcc=40, sr=40.0, win_length_sec=0.3, hop_length_sec=0.2, 
                 maxdim=2, sample=None, thresh=np.inf, alpha=0.1, max_points=500, seed=28):
        """Inicializar la base de datos de diagramas para características MFCC.
        Args:
            labels (list): Lista de identificadores únicos para los diagramas.
            n_mfcc (int): Cantidad de coeficientes MFCC a calcular.
            sr (float): Frecuencia de muestreo de la señal.
            win_length_sec (float): Longitud de ventana en segundos para MFCC.
            hop_length_sec (float): Paso en segundos para MFCC.
            maxdim (int): Dimensión máxima de homología a calcular.
            sample (int, opcional): Si se indica, muestrea aleatoriamente esa cantidad de diagramas.
            thresh (float): Umbral para el complejo de Vietoris-Rips.
            alpha (float): Proporción de subsampling para FPS.
            max_points (int): Número máximo de puntos a conservar tras el subsampling.
        """
        self.labels = labels
        self.maxdim = maxdim
        self.db = {label: {d: [] for d in range(maxdim + 1)} for label in labels}
        self.seed = seed
        self.sample = sample
        self.thresh = thresh
        self.alpha = alpha
        self.max_points = max_points
        
        # Parámetros MFCC
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.win_length = int(win_length_sec * sr)
        self.hop_length = max(1, int(hop_length_sec * sr))
        
        # Validación: librosa requiere win_length >= n_mfcc
        if self.win_length < self.n_mfcc:
            # Ajustar win_length para que sea al menos n_mfcc
            self.win_length = self.n_mfcc
        
        # Generador RNG para muestreo
        self.rng = np.random.default_rng(self.seed)

    def add_signal(self, signal, label):
        """Calcular y agregar el diagrama de persistencia de una señal en la base de datos.
        
        Args:
            signal (np.ndarray): Arreglo 1D de la serie temporal.
            label (int): Identificador único del diagrama.
        
        Returns:
            bool: True si se agregó exitosamente, False si falló.
        """
        diagrams = self.transform(signal)
        
        try:
            # Agregar cada diagrama de persistencia a la base de datos.
            for dim in range(len(diagrams)):
                self.db[label][dim].append(diagrams[dim])
            return True
        except TypeError:
            print(f"Error al agregar la señal con etiqueta {label}: diagrams es None")
            return False
    
    def get_diagrams(self, label, dim=0):
        """Recuperar todos los diagramas de persistencia para una etiqueta y dimensión dadas.
        
        Args:
            label (int): Identificador único del diagrama.
            dim (int): Dimensión de homología.
            sample (int, opcional): Si se indica, muestrea aleatoriamente esa cantidad de diagramas.
        Returns:
            list: Lista de diagramas de persistencia, cada uno como np.ndarray de forma (~N, 2)
        """
        output = self.db[label][dim]  # Mantener como lista ya que los diagramas tienen formas variables
        if self.sample is not None and len(output) > 0:
            indices = self.rng.choice(len(output), size=min(self.sample, len(output)), replace=False)
            output = [output[i] for i in indices]
        return output
    
    def transform(self, signal):
        """Calcular los diagramas de persistencia para una señal usando características MFCC.
        Args:
            signal (np.ndarray): Arreglo 1D de la serie temporal.
        Returns:
            list: Lista de diagramas de persistencia, cada uno como np.ndarray de forma (~N, 2)
        """
        # Verificar el caché primero
        cache = get_cache()
        cached = cache.get_mfcc(
            signal, self.n_mfcc, self.sr, self.win_length, self.hop_length,
            self.maxdim, self.thresh, self.alpha, self.max_points
        )
        if cached is not None:
            return cached
        
        try:
            # Convertir señal a float32
            x = signal.astype(np.float32).copy()
            
            # Calcular características MFCC
            mfcc = librosa.feature.mfcc(
                y=x,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.win_length,
                hop_length=self.hop_length,
                n_mels=20,
                dct_type=2,
                norm='ortho',
                center=False
            )
            
            # Calcular características delta y delta-delta con ancho adaptativo
            width = min(mfcc.shape[1], 3) if mfcc.shape[1] < 9 else 9
            if width % 2 == 0:
                width -= 1
            width = max(3, width)
            
            mfcc_delta = librosa.feature.delta(mfcc, mode='mirror', order=1)
            mfcc_delta_delta = librosa.feature.delta(mfcc, mode='mirror', order=2)
            
            # Concatenar MFCC, delta y delta-delta
            mfcc_combined = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta), axis=0)
            
            # Normalizar
            mfcc_combined_norm = (mfcc_combined - np.mean(mfcc_combined, axis=1, keepdims=True)) / \
                                 (np.std(mfcc_combined, axis=1, keepdims=True) + 1e-8)
            
            # Transponer para obtener forma (frames, emb_dim)
            point_cloud = mfcc_combined_norm.T
            
        except Exception as e:
            print(f"Error al calcular características MFCC: {e}")
            return None
        
        # Aplicar subsampling FPS
        nb_points = max(1, min(int(self.alpha * point_cloud.shape[0]), self.max_points))
        if nb_points < point_cloud.shape[0]:
            point_cloud_sparse = np.array(fps(point_cloud, nb_points=nb_points, starting_point=None))
        else:
            point_cloud_sparse = point_cloud
        
        # Calcular diagramas de persistencia
        diagrams = compute_persistence(point_cloud_sparse, maxdim=self.maxdim, thresh=self.thresh)
        
        # Guardar en el caché
        cache.put_mfcc(
            signal, self.n_mfcc, self.sr, self.win_length, self.hop_length,
            self.maxdim, self.thresh, self.alpha, self.max_points,
            diagrams
        )
        
        return diagrams
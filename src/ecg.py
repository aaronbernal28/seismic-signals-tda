"""
Source:
    Profesor: Gabriel Minian
    Topología aplicada y análisis topológico de datos
    2do Cuatrimestre 2025
    Departamento de Matemática, UBA
    https://mate.dm.uba.ar/~gminian/materias/tda2025/tda.html
"""
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams, wasserstein
import warnings
import os
import wfdb
from scipy import signal
warnings.filterwarnings('ignore')

DURATION = 10
SAMPLE_RATE_REAL = 200
N_POINTS_REAL = DURATION * SAMPLE_RATE_REAL

def download_and_save_mit_bih_record(record_name):
    try:
        record = wfdb.rdrecord(record_name, pn_dir='mitdb', sampto=N_POINTS_REAL)
        ecg_signal = record.p_signal[:, 0]
        
        ecg_signal = ecg_signal - np.mean(ecg_signal)
        ecg_signal = ecg_signal / np.std(ecg_signal)
        
        data_dir = "ecg_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        filename = os.path.join(data_dir, f"{record_name}_ecg.npy")
        np.save(filename, ecg_signal)
        
        return ecg_signal
        
    except Exception as e:
        print(f"Error descargando {record_name}: {e}")
        return None

def load_saved_ecg_data():
    data_dir = "ecg_data"
    if not os.path.exists(data_dir):
        return {}
    
    real_signals = {}
    file_mapping = {
        '100_ecg.npy': 'ECG  Normal',
        '101_ecg.npy': 'ECG  Normal 2',
        '105_ecg.npy': 'ECG  Arritmia', 
        '203_ecg.npy': 'ECG  Fibrilación'
    }
    
    for filename, signal_name in file_mapping.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            ecg_signal = np.load(filepath)
            real_signals[signal_name] = ecg_signal
    
    return real_signals

def load_real_ecg_data():
    real_signals = load_saved_ecg_data()
    
    if not real_signals:
        mit_records = [
            ('100', 'ECG  Normal'),
            ('101', 'ECG  Normal 2'), 
            ('105', 'ECG  Arritmia'),
            ('203', 'ECG  Fibrilación')
        ]
        
        for record_id, record_name in mit_records:
            ecg_data = download_and_save_mit_bih_record(record_id)
            if ecg_data is not None:
                real_signals[record_name] = ecg_data
    
    return real_signals

def takens_embedding(signal, dim=4, delay=10):
    n = len(signal) - (dim - 1) * delay
    embedded = np.zeros((n, dim))
    for i in range(n):
        embedded[i] = signal[i:i + dim * delay:delay]
    return embedded

def compute_persistence(point_cloud, maxdim=1, thresh=np.inf, metric='euclidean'):
    result = ripser(point_cloud, maxdim=maxdim, thresh=thresh, metric=metric)
    return result['dgms']

def bottleneck_distance(dgm1, dgm2):
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    
    n1, n2 = len(dgm1), len(dgm2)
    n = max(n1, n2)
    
    cost_matrix = np.zeros((n, n))
    
    for i in range(n1):
        for j in range(n2):
            cost_matrix[i, j] = max(abs(dgm1[i, 0] - dgm2[j, 0]), abs(dgm1[i, 1] - dgm2[j, 1]))
    
    for i in range(n1):
        diag_proj = (dgm1[i, 0] + dgm1[i, 1]) / 2
        diag_cost = max(abs(dgm1[i, 0] - diag_proj), abs(dgm1[i, 1] - diag_proj))
        for j in range(n2, n):
            cost_matrix[i, j] = diag_cost
    
    for j in range(n2):
        diag_proj = (dgm2[j, 0] + dgm2[j, 1]) / 2
        diag_cost = max(abs(dgm2[j, 0] - diag_proj), abs(dgm2[j, 1] - diag_proj))
        for i in range(n1, n):
            cost_matrix[i, j] = diag_cost
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    return cost_matrix[row_ind, col_ind].max()

def compute_distances(dgm1, dgm2):
    dgm1_finite = dgm1[dgm1[:, 1] != np.inf]
    dgm2_finite = dgm2[dgm2[:, 1] != np.inf]
    
    if len(dgm1_finite) == 0 or len(dgm2_finite) == 0:
        return np.inf, np.inf
    
    d_bottleneck = bottleneck_distance(dgm1_finite, dgm2_finite)
    d_wasserstein = wasserstein(dgm1_finite, dgm2_finite)
    
    return d_bottleneck, d_wasserstein

def analyze_signals(signals, save_suffix=""):
    dim, delay = 4, 10
    
    diagrams = {}
    
    for name, signal in signals.items():
        embedding = takens_embedding(signal, dim, delay)
        dgms = compute_persistence(embedding, maxdim=1)
        diagrams[name] = dgms
    
    fig = plt.figure(figsize=(20, 12))
    
    for idx, (name, ecg_signal) in enumerate(signals.items()):
        ax1 = plt.subplot(len(signals), 2, idx * 2 + 1)
        ax1.plot(ecg_signal, linewidth=1.5, color='steelblue')
        ax1.set_title(f'{name}', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Muestras', fontsize=9)
        ax1.set_ylabel('Amplitud', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(len(signals), 2, idx * 2 + 2)
        plot_diagrams(diagrams[name], show=False, ax=ax2, legend=False)
        ax2.set_title(f'Diagrama de Persistencia', fontsize=11, fontweight='bold')
        ax2.set_xlabel('')
        ax2.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(f'tda_analysis_{save_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    names = list(signals.keys())
    dist_h1_bottleneck = np.zeros((len(names), len(names)))
    dist_h1_wasserstein = np.zeros((len(names), len(names)))
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i < j:
                d_b1, d_w1 = compute_distances(diagrams[name1][1], diagrams[name2][1])
                dist_h1_bottleneck[i, j] = dist_h1_bottleneck[j, i] = d_b1
                dist_h1_wasserstein[i, j] = dist_h1_wasserstein[j, i] = d_w1
    
    print(f"DISTANCIA BOTTLENECK H1")
    print(f"{'':20s} " + " ".join([f"{n:15s}" for n in names]))
    for i, name1 in enumerate(names):
        row = f"{name1:20s} "
        for j in range(len(names)):
            if i == j:
                row += f"{'---':>15s} "
            else:
                row += f"{dist_h1_bottleneck[i, j]:>15.6f} "
        print(row)
    
    print(f"\nDISTANCIA WASSERSTEIN H1")
    print(f"{'':20s} " + " ".join([f"{n:15s}" for n in names]))
    for i, name1 in enumerate(names):
        row = f"{name1:20s} "
        for j in range(len(names)):
            if i == j:
                row += f"{'---':>15s} "
            else:
                row += f"{dist_h1_wasserstein[i, j]:>15.6f} "
        print(row)
    
    return diagrams

def analyze_timeseries():
    real_signals = load_real_ecg_data()
    
    if real_signals:
        print(f"Analizando {len(real_signals)} señales ")
        print(f"Duración: {DURATION} segundos")
        print(f"Puntos por señal: {N_POINTS_REAL}")
        print(f"Frecuencia de muestreo: {SAMPLE_RATE_REAL} Hz")
        
        analyze_signals(real_signals, " ")
    else:
        print("No se pudieron cargar datos reales")

if __name__ == "__main__":
    analyze_timeseries()
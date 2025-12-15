"""
Funciones reutilizables de preprocesamiento y manejo de datos para señales sísmicas.
Estas funciones se emplean en varios scripts para gestionar eventos, generar intervalos y
descargar o preprocesar señales.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from obspy.core.event import Catalog
from obspy import UTCDateTime
import warnings
from config.config import (
    EVENT_INTERVAL_START_OFFSET_MIN,
    EVENT_INTERVAL_START_OFFSET_MAX,
    EVENT_INTERVAL_END_OFFSET_MIN,
    EVENT_INTERVAL_END_OFFSET_MAX,
    NON_EVENT_INTERVAL_DURATION
)


# ============================================================================
# FUNCIONES DE GESTIÓN DE EVENTOS (desde 01_get_raw_events.py)
# ============================================================================

def download_events(client, start_time, end_time, latitude, longitude, window_degrees=5):
    """
    Descargar catálogo de eventos sísmicos desde un cliente FDSN.
    
    Parameters:
    -----------
    client : obspy.clients.fdsn.Client
        Instancia del cliente FDSN
    start_time : obspy.UTCDateTime
        Tiempo inicial para la búsqueda
    end_time : obspy.UTCDateTime
        Tiempo final para la búsqueda
    latitude : float
        Latitud de la estación
    longitude : float
        Longitud de la estación
    window_degrees : float
        Ventana de búsqueda en grados (por defecto: 5)
        
    Returns:
    --------
    obspy.core.event.Catalog : Catálogo descargado
    
    Raises:
    -------
    Exception : Si la descarga falla
    """
    try:
        catalog = client.get_events(
            starttime=start_time,
            endtime=end_time,
            minlatitude=latitude - window_degrees,
            maxlatitude=latitude + window_degrees,
            minlongitude=longitude - window_degrees,
            maxlongitude=longitude + window_degrees
        )
        return catalog
    except Exception as e:
        raise e


def filter_events_by_distance(catalog, station_lat, station_lon, max_distance_km):
    """
    Filtrar eventos en función de la distancia a la estación.
    
    Parameters:
    -----------
    catalog : obspy.core.event.Catalog
        Catálogo de eventos
    station_lat : float
        Latitud de la estación
    station_lon : float
        Longitud de la estación
    max_distance_km : float
        Distancia máxima en kilómetros
        
    Returns:
    --------
    list : Lista de eventos filtrados
    """
    max_distance_degrees = max_distance_km / 111.19
    filtered = []
    
    for event in catalog:
        origin = event.preferred_origin() or event.origins[0]
        eq_lat = origin.latitude
        eq_lon = origin.longitude
        
        distance_degrees = locations2degrees(eq_lat, eq_lon, station_lat, station_lon)
        
        if distance_degrees <= max_distance_degrees:
            filtered.append(event)
    
    return filtered


# ============================================================================
# FUNCIONES DE GENERACIÓN DE INTERVALOS (desde 02_get_intervals.py)
# ============================================================================

def extract_event_data(catalog):
    """
    Extraer información de eventos desde un catálogo de ObsPy.
    
    Parameters:
    -----------
    catalog : obspy.core.event.Catalog
        Catálogo de eventos a procesar
        
    Returns:
    --------
    pd.DataFrame : DataFrame con columnas: date, lat, long, mag, id
    """
    events_data = []
    
    for event in catalog:
        # Obtener origen preferido o el primero disponible
        origin = event.preferred_origin() or event.origins[0]
        
        # Obtener magnitud preferida o la primera lista
        magnitude = event.preferred_magnitude() or event.magnitudes[0] if event.magnitudes else None
        
        # Extraer ID del evento
        event_id = str(event.resource_id).split('/')[-1] if event.resource_id else None
        
        # Construir diccionario del evento
        event_dict = {
            'date': origin.time.datetime,
            'lat': origin.latitude,
            'long': origin.longitude,
            'mag': magnitude.mag if magnitude else None,
            'id': event_id
        }
        
        events_data.append(event_dict)
    
    # Crear DataFrame
    df = pd.DataFrame(events_data)
    
    # Ordenar por fecha
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def create_event_intervals(events_df):
    """
    Crear intervalos temporales para cada evento (label=1) con duraciones aleatorias.
    Offset inicial: uniform(EVENT_INTERVAL_START_OFFSET_MIN, EVENT_INTERVAL_START_OFFSET_MAX) segundos antes del evento
    Offset final: uniform(EVENT_INTERVAL_END_OFFSET_MIN, EVENT_INTERVAL_END_OFFSET_MAX) segundos después del evento
    
    Parameters:
    -----------
    events_df : pd.DataFrame
        DataFrame con datos de eventos que incluye la columna 'date'
        
    Returns:
    --------
    pd.DataFrame : DataFrame con columnas: id, date, start_time, end_time, duration, lat, long, mag, label
    """
    intervals = []
    
    for _, event in events_df.iterrows():
        # Convertir la fecha si viene como texto
        if isinstance(event['date'], str):
            event_time = pd.to_datetime(event['date'])
        else:
            event_time = event['date']
        
        # Calcular los límites del intervalo con offsets aleatorios
        start_offset = np.random.uniform(EVENT_INTERVAL_START_OFFSET_MIN, EVENT_INTERVAL_START_OFFSET_MAX)  # segundos antes del evento
        end_offset = np.random.uniform(EVENT_INTERVAL_END_OFFSET_MIN, EVENT_INTERVAL_END_OFFSET_MAX)  # segundos después del evento
        
        start_time = event_time - timedelta(seconds=start_offset)
        end_time = event_time + timedelta(seconds=end_offset)
        
        # Calcular duración
        duration = (end_time - start_time).total_seconds()
        
        # Crear diccionario del intervalo
        interval_dict = {
            'id': event['id'],
            'date': event_time,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'lat': event['lat'],
            'long': event['long'],
            'mag': event['mag'],
            'label': 1
        }
        
        intervals.append(interval_dict)
    
    # Crear DataFrame
    df = pd.DataFrame(intervals)
    
    return df


def intervals_overlap(start1, end1, start2, end2):
    """
    Comprobar si dos intervalos temporales se superponen.
    
    Parameters:
    -----------
    start1, end1 : datetime
        Límites del primer intervalo
    start2, end2 : datetime
        Límites del segundo intervalo
        
    Returns:
    --------
    bool : True si se superponen, False en caso contrario
    """
    return start1 <= end2 and start2 <= end1


def generate_non_event_intervals(event_intervals_df, global_start, global_end, num_intervals=None):
    """
    Generar intervalos sin evento (label=0) sin superposición.
    Cada intervalo tiene una duración aleatoria que sigue la distribución de los eventos.
    
    Parameters:
    -----------
    event_intervals_df : pd.DataFrame
        DataFrame con los intervalos de eventos
    global_start : datetime
        Inicio del rango global
    global_end : datetime
        Fin del rango global
    num_intervals : int, optional
        Cantidad de intervalos sin evento. Si es None, iguala la cantidad de eventos.
        
    Returns:
    --------
    pd.DataFrame : DataFrame con los intervalos generados
    """
    if num_intervals is None:
        num_intervals = len(event_intervals_df)
    
    # Convertir a datetime
    global_start = pd.to_datetime(global_start)
    global_end = pd.to_datetime(global_end)
    
    # Convertir intervalos de eventos a datetime
    event_starts = pd.to_datetime(event_intervals_df['start_time'])
    event_ends = pd.to_datetime(event_intervals_df['end_time'])
    
    non_event_intervals = []
    max_attempts = num_intervals * 100  # Prevenir bucle infinito
    attempts = 0
    
    print(f"Generando {num_intervals} intervalos sin evento con duraciones aleatorias...")
    
    while len(non_event_intervals) < num_intervals and attempts < max_attempts:
        attempts += 1
        
        # Generar duración aleatoria análoga a los intervalos de eventos
        start_offset = np.random.uniform(EVENT_INTERVAL_START_OFFSET_MIN, EVENT_INTERVAL_START_OFFSET_MAX)
        end_offset = np.random.uniform(EVENT_INTERVAL_END_OFFSET_MIN, EVENT_INTERVAL_END_OFFSET_MAX)
        duration = start_offset + end_offset  # Duración total
        
        # Asegurar duración mínima de 40 segundos
        if duration < 40:
            duration = 40
        
        # Generar inicio aleatorio
        time_range_seconds = (global_end - global_start).total_seconds()
        random_offset = np.random.uniform(0, time_range_seconds - duration)
        random_start = global_start + timedelta(seconds=random_offset)
        random_end = random_start + timedelta(seconds=duration)
        
        # Verificar si el intervalo se superpone con algún evento
        overlaps = False
        for i in range(len(event_intervals_df)):
            if intervals_overlap(random_start, random_end, event_starts.iloc[i], event_ends.iloc[i]):
                overlaps = True
                break
        
        # Si no hay superposición, agregar a los intervalos sin evento
        if not overlaps:
            interval_dict = {
                'id': f"non_event_{len(non_event_intervals) + 1}",
                'date': random_start + timedelta(seconds=duration/2),  # Medio del intervalo
                'start_time': random_start,
                'end_time': random_end,
                'duration': duration,
                'lat': None,
                'long': None,
                'mag': None,
                'label': 0
            }
            non_event_intervals.append(interval_dict)
            
            if len(non_event_intervals) % 10 == 0:
                print(f"  Generados {len(non_event_intervals)}/{num_intervals} intervalos sin evento")
    
    if len(non_event_intervals) < num_intervals:
        print(f"⚠ Advertencia: Solo se generaron {len(non_event_intervals)} intervalos no superpuestos")
    
    df = pd.DataFrame(non_event_intervals)
    return df


# ============================================================================
# FUNCIONES DE PROCESAMIENTO DE SEÑALES (desde 03_get_signals.py)
# ============================================================================

def preprocess_trace(trace):
    """
    Aplicar preprocesamiento básico a una traza sísmica.
    
    Steps:
    1. Detrend - remove mean
    2. Detrend - remove linear trend
    3. Remove instrument response (convert to displacement)
    
    Parameters:
    -----------
    trace : obspy.Trace
        Input seismic trace
        
    Returns:
    --------
    tuple: (preprocessed_trace, success_flag)
        - preprocessed_trace: obspy.Trace - Preprocessed trace
        - success_flag: bool - True if all steps succeeded, False otherwise
    """
    # Copia la traza para no modificar la original
    trace_copy = trace.copy()
    preprocessing_success = True
    
    try:
        # Step 1: Eliminar media
        trace_copy.detrend('demean')
        
        # Step 2: eliminar tendencia lineal
        trace_copy.detrend('linear')
        
    except Exception as e:
        print(f"    ⚠ Advertencia: Falló el detrending para la traza {trace.id}: {str(e)[:80]}")
        preprocessing_success = False
        return trace, False  # Devolver original si falla el detrending básico
    
    # Paso 3: Eliminar respuesta del instrumento (convertir a desplazamiento)
    # Este paso es el que más probablemente falle si falta metadata
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace_copy.remove_response(output='DISP')
    except Exception as e:
        print(f"    ⚠ Advertencia: Falló remove_response para la traza {trace.id}: {str(e)[:80]}")
        print(f"       Continuando solo con datos detrended (no convertidos a unidades físicas)")
        preprocessing_success = False
        # Devolver la traza detrended incluso si falla la eliminación de respuesta
    
    return trace_copy, preprocessing_success


def download_waveforms(client, network, station, channel, start_time, end_time):
    """
    Descargar trazas sísmicas desde un cliente FDSN.
    
    Parameters:
    -----------
    client : obspy.clients.fdsn.Client
        Instancia del cliente FDSN
    network : str
        Código de red
    station : str
        Código de estación
    channel : str
        Código de canal
    start_time : obspy.UTCDateTime
        Tiempo inicial
    end_time : obspy.UTCDateTime
        Tiempo final
        
    Returns:
    --------
    obspy.Stream : Stream descargado
    
    Raises:
    -------
    Exception : Si la descarga falla
    """
    try:
        st = client.get_waveforms(
            network=network,
            station=station,
            location="*",
            channel=channel,
            starttime=start_time,
            endtime=end_time
        )
        return st
    except Exception as e:
        raise e


def split_data(intervals_df, train_ratio=0.8, test_ratio=0.2, eval_ratio=0.0, create_eval=False, random_state=42):
    """
    Dividir los datos en conjuntos train, test y opcionalmente eval equilibrando etiquetas.
    
    Parameters:
    -----------
    intervals_df : pd.DataFrame
        DataFrame con la información de intervalos
    train_ratio : float
        Proporción para entrenamiento (por defecto: 0.8)
    test_ratio : float
        Proporción para prueba (por defecto: 0.2)
    eval_ratio : float
        Proporción para evaluación (por defecto: 0.0)
    create_eval : bool
        Si se crea conjunto de evaluación (por defecto: False)
    random_state : int
        Semilla para reproducibilidad (por defecto: 42)
    
    Returns:
    --------
    dict : Diccionario con claves 'train', 'test' y opcionalmente 'eval' que contienen DataFrames
    """
    from sklearn.model_selection import train_test_split
    
    if not create_eval:
        # Ajustar proporciones si no hay conjunto eval
        test_ratio_adjusted = test_ratio / (train_ratio + test_ratio)
        
        # Dividir por etiqueta para conservar equilibrio
        train_data = []
        test_data = []
        
        for label in intervals_df['label'].unique():
            label_data = intervals_df[intervals_df['label'] == label]
            train_label, test_label = train_test_split(
                label_data, test_size=test_ratio_adjusted, random_state=random_state
            )
            train_data.append(train_label)
            test_data.append(test_label)
        
        return {
            'train': pd.concat(train_data).reset_index(drop=True),
            'test': pd.concat(test_data).reset_index(drop=True)
        }
    else:
        # División en tres bloques
        test_eval_ratio = test_ratio + eval_ratio
        eval_ratio_adjusted = eval_ratio / test_eval_ratio
        
        train_data = []
        test_data = []
        eval_data = []
        
        for label in intervals_df['label'].unique():
            label_data = intervals_df[intervals_df['label'] == label]
            
            # Primera división: train vs (test + eval)
            train_label, test_eval_label = train_test_split(
                label_data, test_size=test_eval_ratio, random_state=random_state
            )
            
            # Segunda división: test vs eval
            test_label, eval_label = train_test_split(
                test_eval_label, test_size=eval_ratio_adjusted, random_state=random_state
            )
            
            train_data.append(train_label)
            test_data.append(test_label)
            eval_data.append(eval_label)
        
        return {
            'train': pd.concat(train_data).reset_index(drop=True),
            'test': pd.concat(test_data).reset_index(drop=True),
            'eval': pd.concat(eval_data).reset_index(drop=True)
        }

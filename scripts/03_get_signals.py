"""
Script para obtener formas de onda sísmicas para cada intervalo y almacenar en formato HDF5.
Lee intervalos desde data/processed/intervals.csv y guarda señales en data/processed/signals.hdf5
"""

import sys
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# Agregar directorio padre al path para permitir importaciones desde config y src
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import PROCESSED_DATA_PATH, DATA_CENTER, STATION_CODE, NETWORK, CHANNEL, LATITUDE, LONGITUDE
from src.preprocess import preprocess_trace, download_waveforms, split_data


def fetch_and_store_signals(intervals_df, output_file, split_name="", batch_size=50, max_workers=3):
    """
    Obtener formas de onda para cada intervalo y almacenar en archivo HDF5.
    Usa descarga paralela con lotes más pequeños para eficiencia.
    
    Parámetros:
    -----------
    intervals_df : pd.DataFrame
        DataFrame con información de intervalos incluyendo start_time y end_time
    output_file : Path
        Ruta al archivo HDF5 de salida
    split_name : str
        Nombre del conjunto (p. ej., "train", "test", "eval") para logging
    batch_size : int
        Número de intervalos a procesar en paralelo (por defecto: 50)
    max_workers : int
        Número máximo de workers de descarga paralela (por defecto: 3)
    """
    print(f"\nProcesando conjunto {split_name.upper()}..." if split_name else "Procesando intervalos...")
    
    # Crear archivo HDF5
    with h5py.File(output_file, 'w') as hf:
        # Almacenar metadatos
        metadata_group = hf.create_group('metadata')
        
        # Convertir DataFrame para almacenar en HDF5
        for col in intervals_df.columns:
            if col in ['start_time', 'end_time', 'date']:
                # Convertir datetime a string para almacenamiento
                metadata_group.create_dataset(col, data=intervals_df[col].astype(str).values.astype('S'))
            else:
                # Manejar otras columnas
                col_data = intervals_df[col].values
                if col_data.dtype == 'object':
                    # Convertir object a string
                    col_data = np.array([str(x) if x is not None else '' for x in col_data], dtype='S')
                metadata_group.create_dataset(col, data=col_data)
        
        # Crear grupo de señales
        signals_group = hf.create_group('signals')
        
        total_intervals = len(intervals_df)
        successful = 0
        failed = 0
        
        print(f"Obteniendo formas de onda para {total_intervals} intervalos...")
        print(f"Usando descarga paralela con {max_workers} workers, tamaño de lote: {batch_size}")
        print("-" * 60)
        
        # Procesar en lotes paralelos
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def download_single_interval(row):
            """Descargar forma de onda de un intervalo individual."""
            interval_id = row['id']
            start_time = UTCDateTime(pd.to_datetime(row['start_time']))
            end_time = UTCDateTime(pd.to_datetime(row['end_time']))
            label = row['label']
            
            try:
                client = Client(DATA_CENTER)
                st = download_waveforms(client, NETWORK, STATION_CODE, CHANNEL, start_time, end_time)
                return interval_id, label, st, None
            except Exception as e:
                return interval_id, label, None, str(e)
        
        # Procesar en lotes
        for batch_start in range(0, total_intervals, batch_size):
            batch_end = min(batch_start + batch_size, total_intervals)
            batch_df = intervals_df.iloc[batch_start:batch_end]
            
            print(f"\nProcesando lote {batch_start+1}-{batch_end} de {total_intervals}...")
            
            # Enviar descargas del lote
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_row = {
                    executor.submit(download_single_interval, row): (idx, row)
                    for idx, row in batch_df.iterrows()
                }
                
                # Procesar descargas completadas
                for future in as_completed(future_to_row):
                    idx, row = future_to_row[future]
                    try:
                        interval_id, label, st, error = future.result()
                        
                        if st is not None and len(st) > 0:
                            # Almacenar en HDF5
                            interval_group = signals_group.create_group(str(interval_id))
                            
                            for trace_idx, trace in enumerate(st):
                                # Aplicar preprocesamiento a la traza
                                preprocessed_trace, preprocess_success = preprocess_trace(trace)
                                
                                # Almacenar datos de traza preprocesada
                                trace_group = interval_group.create_group(f'trace_{trace_idx}')
                                trace_group.create_dataset('data', data=preprocessed_trace.data, compression='gzip')
                                
                                # Almacenar metadatos de traza
                                trace_group.attrs['sampling_rate'] = preprocessed_trace.stats.sampling_rate
                                trace_group.attrs['npts'] = preprocessed_trace.stats.npts
                                trace_group.attrs['network'] = preprocessed_trace.stats.network
                                trace_group.attrs['station'] = preprocessed_trace.stats.station
                                trace_group.attrs['location'] = preprocessed_trace.stats.location
                                trace_group.attrs['channel'] = preprocessed_trace.stats.channel
                                trace_group.attrs['starttime'] = str(preprocessed_trace.stats.starttime)
                                trace_group.attrs['endtime'] = str(preprocessed_trace.stats.endtime)
                                trace_group.attrs['preprocessed'] = preprocess_success  # Bandera indica si TODOS los pasos de preprocesamiento tuvieron éxito
                                trace_group.attrs['detrended'] = True  # El detrending básico siempre se aplica
                                trace_group.attrs['response_removed'] = preprocess_success  # La remoción de respuesta solo tiene éxito si los metadatos están disponibles
                            
                            # Almacenar metadatos a nivel de intervalo
                            interval_group.attrs['label'] = label
                            interval_group.attrs['num_traces'] = len(st)
                            interval_group.attrs['interval_id'] = str(interval_id)
                            
                            successful += 1
                            print(f"  ✓ [{successful+failed}/{total_intervals}] Intervalo {interval_id}: {len(st)} traza(s)")
                        else:
                            failed += 1
                            error_msg = error[:80] + "..." if error and len(error) > 80 else error
                            print(f"  ✗ [{successful+failed}/{total_intervals}] Intervalo {interval_id}: {error_msg or 'Sin datos'}")
                    
                    except Exception as e:
                        failed += 1
                        print(f"  ✗ [{successful+failed}/{total_intervals}] Error procesando: {str(e)[:80]}")
            
            # Actualización de progreso después de cada lote
            print(f"\nLote completo. Progreso: {batch_end}/{total_intervals}")
            print(f"  Exitosos: {successful}, Fallidos: {failed}, Tasa: {successful/(successful+failed)*100:.1f}%")
            print("-" * 60)
        
        print(f"\n{'=' * 60}")
        print(f"Resumen:")
        print(f"  Intervalos totales: {total_intervals}")
        print(f"  Exitosos: {successful}")
        print(f"  Fallidos: {failed}")
        print(f"  Tasa de éxito: {successful/total_intervals*100:.1f}%")
        print(f"{'=' * 60}")


def main():
    """Función principal de ejecución."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Obtener y dividir señales sísmicas en conjuntos train/test/eval')
    parser.add_argument('--create-eval', action='store_true', default=False,
                        help='Crear conjunto de evaluación (por defecto: False)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Proporción del conjunto de entrenamiento (por defecto: 0.8)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='Proporción del conjunto de prueba (por defecto: 0.2)')
    parser.add_argument('--eval-ratio', type=float, default=0.0,
                        help='Proporción del conjunto de evaluación (por defecto: 0.0)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Número de intervalos a procesar en paralelo (por defecto: 50)')
    parser.add_argument('--max-workers', type=int, default=3,
                        help='Máximo de workers paralelos de descarga (por defecto: 3)')

    args = parser.parse_args()
    
    print("=" * 60)
    print("Script de Muestreo de Señales con División Train/Test/Eval")
    print("=" * 60)
    
    # Definir rutas de archivos
    input_file = Path(PROCESSED_DATA_PATH) / "signal_intervals.csv"
    
    # Verificar si el archivo de entrada existe
    if not input_file.exists():
        print(f"✗ Error: Archivo de entrada no encontrado: {input_file}")
        print("  Por favor asegúrate de que signal_intervals.csv existe en data/processed/")
        print("  (Ejecuta get_intervals.py o get_signal_intervals.py primero)")
        return
    
    print(f"Leyendo intervalos desde: {input_file}")
    
    try:
        # Leer CSV de intervalos
        intervals_df = pd.read_csv(input_file)
        print(f"✓ Cargados {len(intervals_df)} intervalos")
        print(f"  Columnas: {list(intervals_df.columns)}")
        print(f"  Etiquetas: {intervals_df['label'].value_counts().to_dict()}")
        
        # Dividir datos
        print(f"\nDividiendo datos (train={args.train_ratio}, test={args.test_ratio}" + 
              (f", eval={args.eval_ratio}" if args.create_eval else "") + ")...")
        splits = split_data(
            intervals_df, 
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            eval_ratio=args.eval_ratio,
            create_eval=args.create_eval
        )
        
        for split_name, split_df in splits.items():
            print(f"\n{split_name.upper()} conjunto: {len(split_df)} intervalos")
            print(f"  Etiquetas: {split_df['label'].value_counts().to_dict()}")
        
        # Crear directorio de salida si no existe
        output_dir = Path(PROCESSED_DATA_PATH)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Obtener y almacenar señales para cada conjunto
        total_size = 0
        
        for split_name, split_df in splits.items():
            output_file = output_dir / f"signals_{split_name}.hdf5"
            print(f"\n{'=' * 60}")
            print(f"Procesando conjunto {split_name.upper()}")
            print(f"{'=' * 60}")
            
            fetch_and_store_signals(
                split_df, 
                output_file, 
                split_name, 
                batch_size=args.batch_size,
                max_workers=args.max_workers
            )
            
            file_size_mb = output_file.stat().st_size / (1024*1024)
            total_size += file_size_mb
            print(f"\n✓ Señales {split_name.upper()} guardadas en: {output_file}")
            print(f"  Tamaño del archivo: {file_size_mb:.2f} MB")
        
        print("\n" + "=" * 60)
        print("✓ Todo el muestreo de señales completado!")
        print(f"  Tamaño total: {total_size:.2f} MB")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
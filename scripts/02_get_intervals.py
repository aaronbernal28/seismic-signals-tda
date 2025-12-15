"""
Script para generar intervalos de tiempo para eventos sísmicos y no eventos.
Lee datos de eventos crudos y crea signal_intervals.csv con:
- Intervalos de eventos (label=1): Ventanas aleatorias alrededor de cada evento
- Intervalos de no eventos (label=0): Ventanas aleatorias de 150s donde no ocurren eventos
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from obspy import read_events

# Agregar directorio padre al path para permitir importaciones desde config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, START_TIME, END_TIME
from config.config import (
    EVENT_INTERVAL_START_OFFSET_MIN, EVENT_INTERVAL_START_OFFSET_MAX,
    EVENT_INTERVAL_END_OFFSET_MIN, EVENT_INTERVAL_END_OFFSET_MAX
)
from src.preprocess import extract_event_data, create_event_intervals, generate_non_event_intervals


def main():
    """Función principal de ejecución."""
    print("=" * 60)
    print("Script de Generación de Intervalos de Señales")
    print("=" * 60)
    
    # Fijar semilla para reproducibilidad
    np.random.seed(28)

    # Definir rutas de archivos
    input_file = Path(RAW_DATA_PATH) / "events.xml"
    output_file = Path(PROCESSED_DATA_PATH) / "signal_intervals.csv"
    
    # Verificar si el archivo de entrada existe
    if not input_file.exists():
        print(f"✗ Error: Archivo de entrada no encontrado: {input_file}")
        print("  Por favor ejecuta get_raw_data.py primero para descargar eventos.")
        return
    
    print(f"Leyendo eventos desde: {input_file}")
    
    try:
        # Leer el catálogo de eventos desde XML
        catalog = read_events(str(input_file))
        print(f"✓ Catálogo cargado con {len(catalog)} eventos")
        
        # Extraer datos de eventos a DataFrame
        print("Extrayendo datos de eventos...")
        events_df = extract_event_data(catalog)
        print(f"✓ Extraídos {len(events_df)} eventos")
        
        # Crear intervalos de eventos (label=1)
        print("\nGenerando intervalos de eventos (label=1)...")
        print(f"  Offset inicial: uniforme({EVENT_INTERVAL_START_OFFSET_MIN}, {EVENT_INTERVAL_START_OFFSET_MAX}) segundos antes del evento")
        print(f"  Offset final: uniforme({EVENT_INTERVAL_END_OFFSET_MIN}, {EVENT_INTERVAL_END_OFFSET_MAX}) segundos después del evento")
        event_intervals_df = create_event_intervals(events_df)
        print(f"✓ Creados {len(event_intervals_df)} intervalos de eventos")
        print(f"  Rango de duración: {event_intervals_df['duration'].min():.1f}s a {event_intervals_df['duration'].max():.1f}s")
        print(f"  Duración promedio: {event_intervals_df['duration'].mean():.1f}s")
        
        # Generar intervalos de no eventos (label=0)
        print("\nGenerando intervalos de no eventos (label=0)...")
        non_event_intervals_df = generate_non_event_intervals(
            event_intervals_df,
            global_start=START_TIME.datetime,
            global_end=END_TIME.datetime
        )
        print(f"✓ Creados {len(non_event_intervals_df)} intervalos de no eventos")
        print(f"  Rango de duración: {non_event_intervals_df['duration'].min():.1f}s a {non_event_intervals_df['duration'].max():.1f}s")
        print(f"  Duración promedio: {non_event_intervals_df['duration'].mean():.1f}s")
        
        # Combinar ambos tipos de intervalos
        print("\nCombinando todos los intervalos...")
        all_intervals_df = pd.concat([event_intervals_df, non_event_intervals_df], ignore_index=True)
        
        # Ordenar por tiempo de inicio
        all_intervals_df = all_intervals_df.sort_values('start_time').reset_index(drop=True)
        
        # Crear directorio de salida si no existe
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar a CSV
        all_intervals_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Intervalos de señales guardados en: {output_file}")
        print(f"\nInformación del DataFrame:")
        print(f"  Filas totales: {len(all_intervals_df)}")
        print(f"  Intervalos de eventos (label=1): {len(all_intervals_df[all_intervals_df['label'] == 1])}")
        print(f"  Intervalos de no eventos (label=0): {len(all_intervals_df[all_intervals_df['label'] == 0])}")
        print(f"  Columnas: {list(all_intervals_df.columns)}")
        
        print(f"\nEstadísticas de duración:")
        print(f"  Rango de duración general: {all_intervals_df['duration'].min():.1f}s a {all_intervals_df['duration'].max():.1f}s")
        print(f"  Duración promedio general: {all_intervals_df['duration'].mean():.1f}s")
        
        print(f"\nPrimeros intervalos:")
        print(all_intervals_df.head(10))
        
        print(f"\nDistribución de etiquetas:")
        print(all_intervals_df['label'].value_counts().sort_index())
        
        print("\n" + "=" * 60)
        print("✓ Generación de intervalos completada exitosamente!")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Error generando intervalos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

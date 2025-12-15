"""
Script para descargar datos sísmicos crudos y catálogos de eventos.
Basado en el notebook 01_experiments.ipynb
"""

import sys
from pathlib import Path

# Agregar directorio padre al path para permitir importaciones desde config
sys.path.insert(0, str(Path(__file__).parent.parent))

from obspy.clients.fdsn import Client
from obspy.core.event import Catalog
from config.config import *
from src.preprocess import download_events as download_events_func, filter_events_by_distance


def download_events():
    """Descargar catálogo de eventos sísmicos."""
    print("\nBuscando eventos sísmicos...")
    client = Client(DATA_CENTER)
    
    window = 5  # grados
    
    try:
        catalog = download_events_func(
            client=client,
            start_time=START_TIME,
            end_time=END_TIME,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            window_degrees=window
        )
        
        print(f"  Encontrados {len(catalog)} eventos")
        
        # Filtrar por distancia
        filtered_catalog = filter_events_by_distance(catalog, LATITUDE, LONGITUDE, MAX_DISTANCE_KM)
        
        # Guardar en archivo
        output_file = Path(RAW_DATA_PATH) / "events.xml"
        filtered_catalog_obj = Catalog(events=filtered_catalog)
        filtered_catalog_obj.write(str(output_file), format='QUAKEML')
        
        print(f"✓ Catálogo de eventos guardado en {output_file}")
        print(f"  {len(filtered_catalog)} eventos después del filtrado por distancia")
        return filtered_catalog_obj
        
    except Exception as e:
        print(f"✗ Error descargando eventos: {e}")
        return None


def main():
    """Función principal de ejecución."""
    print("=" * 60)
    print("Script de Adquisición de Datos Sísmicos")
    print("=" * 60)
    print(f"Estación: {NETWORK}.{STATION_CODE}")
    print(f"Rango temporal: {START_TIME} a {END_TIME}")
    print(f"Canal: {CHANNEL}")
    print("=" * 60)
    
    # Descargar datos
    events = download_events()
    
    print("\n" + "=" * 60)
    if events:
        print("✓ Adquisición de datos completada exitosamente!")
    else:
        print("⚠ Adquisición de datos completada con errores")
    print("=" * 60)


if __name__ == "__main__":
    main()

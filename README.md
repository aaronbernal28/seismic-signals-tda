# seismic-signals-tda

**Materia:** Topología aplicada y análisis topológico de datos (Applied Topology and Topological Data Analysis)  
**Profesor:** Dr. Gabriel Minian  
**Institución:** Departamento de Matemática, Universidad de Buenos Aires (UBA)  
**Cuatrimestre:** 2.º cuatrimestre 2025

## Descripción general

Este trabajo presenta una metodología para la detección automática de sismos utilizando Análisis Topológico de Datos (TDA), aplicada a registros de actividad sismológica (a 40Hz) de la Red Sismológica Nacional de Chile. El enfoque procesa las señales mediante Takens embeddings, submuestreo FPS y homología persistente para generar representaciones topológicas robustas frente al ruido. Los experimentos realizados, que incluyen una búsqueda de hiperparámetros y validación cruzada, demuestran que el modelo logra un alto desempeño discriminante en métricas de evaluación como AUCROC.

## Estructura del proyecto

```text
seismic-signals-tda/
├── config/              # Archivos de configuración y parámetros
├── data/
│   ├── raw/             # Datos sísmicos crudos (XML)
│   ├── processed/       # Señales preprocesadas en HDF5
│   └── results/         # CSVs y gráficas generadas
├── notebooks/           # Exploración y prototipos
├── scripts/             # Pipeline completo
│   ├── 01_get_raw_events.py    # Catálogo desde IRIS
│   ├── 02_get_intervals.py     # Intervalos evento/no evento
│   ├── 03_get_signals.py       # Descarga de formas de onda
│   ├── 06_best_model.py        # Evalúa el mejor modelo TE
│   ├── 07_diagnose_empty_diagrams.py # Diagnóstico de diagramas vacíos
│   ├── 08_distance_graph.py    # Grafo k-NN con distancias bottleneck
│   └── 09_epsilon_train.py     # Distribución de diámetros (dim/tau)
├── src/                # Código fuente (models, preprocess, utils)
└── run_grid_search.py  # Búsqueda de hiperparámetros
```

## Metodología

1. **Adquisición de datos**: Descarga datos de eventos sísmicos desde IRIS usando ObsPy.
2. **Procesamiento de señales**: Extrae y preprocesa formas de onda sísmicas.
3. **Extracción de características topológicas**:
   - Aplica la incrustación de Takens a las series temporales.
   - Calcula diagramas de persistencia usando Ripser.
   - Usa Farthest Point Sampling (FPS) para reducir la nube de puntos.
4. **Clasificación**: Entrena clasificadores binarios usando distancias entre diagramas de persistencia (distancias de Wasserstein y Bottleneck).

## Dependencias clave

- **Sismología**: ObsPy para manejo de datos sísmicos
- **Librerías de TDA**: Ripser, Persim, GUDHI, Giotto-TDA
- **Computación científica**: NumPy, Pandas, SciPy
- **Aprendizaje automático**: Scikit-learn
- **Almacenamiento de datos**: HDF5 (h5py)

## Instalación
Se recomienda usar Python 3.12.9.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Flujo de uso

1) Descargar y preparar datos (requiere conexión a IRIS):
```bash
python scripts/01_get_raw_events.py
python scripts/02_get_intervals.py
python scripts/03_get_signals.py
```

2. Realiza la búsqueda de hiperparámetros para entrenar el modelo:
   ```bash
   python run_grid_search.py
   ```

3) Evaluar el mejor modelo y generar salidas numeradas en data/results/best_model_analysis:
```bash
python scripts/06_best_model.py
```

4) Análisis adicionales (opcional):
- Diagnóstico de diagramas vacíos y distribuciones de probabilidad:
  ```bash
  python scripts/07_diagnose_empty_diagrams.py
  ```
- Grafo de distancias bottleneck sobre el set de prueba:
  ```bash
  python scripts/08_distance_graph.py
  ```
- Distribución de diámetros para distintas configuraciones (train):
  ```bash
  python scripts/09_epsilon_train.py
  ```

## Referencias

Material de la materia: https://mate.dm.uba.ar/~gminian/materias/tda2025/tda.html
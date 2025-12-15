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
│   ├── raw/             # Datos sísmicos crudos (formato XML)
│   ├── processed/       # Señales preprocesadas en formato HDF5
│   └── results/         # Resultados de modelos, CSVs y gráficas
├── docs/                # Documentación
├── notebooks/           # Notebooks de Jupyter para experimentos y visualización
├── scripts/             # Pipeline de procesamiento y entrenamiento
│   ├── 01_get_raw_events.py    # Descarga el catálogo desde IRIS
│   ├── 02_get_intervals.py     # Genera intervalos de evento/no evento
│   ├── 03_get_signals.py       # Descarga formas de onda
│   └── 06_best_model.py        # Evalúa la configuración óptima del modelo
├── src/
│   ├── databases.py     # Implementación de base de datos de diagramas de persistencia
│   ├── preprocess.py    # Utilidades de preprocesamiento de señales
│   ├── models/          # Modelos de clasificación
│   └── utils.py         # Funciones utilitarias generales
└── run_grid_search.py   # Script de ajuste de hiperparámetros
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

## Uso

1. Ejecuta los scripts de adquisición de datos en orden:
   ```bash
   python scripts/01_get_raw_events.py
   python scripts/02_get_intervals.py
   python scripts/03_get_signals.py
   ```

2. Realiza la búsqueda de hiperparámetros para entrenar el modelo:
   ```bash
   python run_grid_search.py
   ```

3. Entrena y evalúa el mejor modelo: para reproducir los mejores resultados usando los parámetros encontrados:
   ```bash
   python scripts/06_best_model.py
   ```
Los resultados se guardarán en data/results/best_model_analysis/

## Referencias

Material de la materia: https://mate.dm.uba.ar/~gminian/materias/tda2025/tda.html
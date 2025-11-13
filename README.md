# seismic-signals-tda

**Course:** Topología aplicada y análisis topológico de datos (Applied Topology and Topological Data Analysis)  
**Professor:** Dr. Gabriel Minian  
**Institution:** Departamento de Matemática, Universidad de Buenos Aires (UBA)  
**Semester:** 2nd Semester 2025

## Overview

This project applies Topological Data Analysis (TDA) techniques to seismic signal classification. The approach uses persistence diagrams derived from Takens embeddings of time series data to distinguish between seismic events and non-events.

## Project Structure

```
seismic-signals-tda/
├── config/              # Configuration files and parameters
├── data/
│   ├── raw/            # Raw seismic event data (XML format)
│   └── processed/      # Preprocessed signals in HDF5 format
├── notebooks/          # Jupyter notebooks for experiments and analysis
├── scripts/            # Data acquisition and preprocessing scripts
└── src/
    ├── databases.py    # Persistence diagram database implementation
    ├── preprocess.py   # Signal preprocessing utilities
    ├── utils.py        # General utility functions
    └── models/         # Classification models using TDA features
```

## Methodology

1. **Data Acquisition**: Download seismic event data from IRIS using ObsPy
2. **Signal Processing**: Extract and preprocess seismic waveforms
3. **Topological Feature Extraction**:
   - Apply Takens embedding to time series signals
   - Compute persistence diagrams using Ripser
   - Use Farthest Point Sampling (FPS) for point cloud reduction
4. **Classification**: Train binary classifiers using distances between persistence diagrams (Wasserstein and Bottleneck distances)

## Key Dependencies

- **Seismology**: ObsPy for seismic data handling
- **TDA Libraries**: Ripser, Persim, GUDHI, Giotto-TDA
- **Scientific Computing**: NumPy, Pandas, SciPy
- **Machine Learning**: Scikit-learn
- **Data Storage**: HDF5 (h5py)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run data acquisition scripts in order:
   ```bash
   python scripts/01_get_raw_events.py
   python scripts/02_get_intervals.py
   python scripts/03_get_signals.py
   ```

2. Explore notebooks for analysis and model training:
   - `07_model2.ipynb`: Binary classification using persistence diagram databases

## References

Course materials: https://mate.dm.uba.ar/~gminian/materias/tda2025/tda.html
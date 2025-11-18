# seismic-signals-tda

**Course:** Topología aplicada y análisis topológico de datos (Applied Topology and Topological Data Analysis)  
**Professor:** Dr. Gabriel Minian  
**Institution:** Departamento de Matemática, Universidad de Buenos Aires (UBA)  
**Semester:** 2nd Semester 2025

## Overview

This project applies Topological Data Analysis (TDA) techniques to seismic signal classification. The approach uses persistence diagrams derived from Takens embeddings of time series data to distinguish between seismic events and non-events.

## Project Structure

```text
seismic-signals-tda/
├── config/              # Configuration files and parameters
├── data/
│   ├── raw/             # Raw seismic event data (XML format)
│   ├── processed/       # Preprocessed signals in HDF5 format
│   └── results/         # Model results, CSVs, and plots
├── docs/                # Documentation
├── notebooks/           # Jupyter notebooks for experiments and visualization
├── scripts/             # Processing and Training Pipeline
│   ├── 01_get_raw_events.py    # Download catalog from IRIS
│   ├── 02_get_intervals.py     # Generate event/non-event intervals
│   ├── 03_get_signals.py       # Download waveforms
│   ├── 04_run_model2.py        # Train Takens Embedding model
│   ├── 05_run_model3.py        # Train MFCC model
│   └── 06_best_model.py        # Evaluate the optimal model configuration
├── src/
│   ├── databases.py     # Persistence diagram database implementation
│   ├── preprocess.py    # Signal preprocessing utilities
│   ├── models/          # Classification models (TE and MFCC variants)
│   └── utils.py         # General utility functions
└── run_grid_search.py   # Hyperparameter tuning script
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

2. Train & Evaluate Best Model: To reproduce the best results (~0.92 AUCROC) using the identified parameters:
   ```bash
   python scripts/06_best_model.py
   ```
Results will be saved to data/results/best_model_analysis/

## Colab
Run the following code in a Google Colab notebook to set up the environment and execute the grid search for model training:
```python
# Clone the repository
!git clone https://github.com/aaronbernal28/seismic-signals-tda.git

# Install dependencies (assuming there's a requirements.txt file in the repo)
!pip install -r seismic-signals-tda/requirements.txt

# Change to the repository directory
%cd seismic-signals-tda

# Run the evaluation script
!python scripts/06_best_model.py
```

## References

Course materials: https://mate.dm.uba.ar/~gminian/materias/tda2025/tda.html
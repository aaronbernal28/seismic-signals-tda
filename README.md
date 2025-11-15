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
│   ├── processed/      # Preprocessed signals in HDF5 format
│   └── results/        # Model results and outputs
├── docs/               # Documentation
├── notebooks/          # Jupyter notebooks for experiments and analysis
├── scripts/            # Data acquisition and preprocessing scripts
│   └── fps/           # Farthest Point Sampling examples
├── src/
│   ├── databases.py    # Persistence diagram database implementation
│   ├── preprocess.py   # Signal preprocessing utilities
│   ├── utils.py        # General utility functions
│   ├── ecg.py         # ECG signal utilities
│   └── models/         # Classification models using TDA features
│       ├── model1.py
│       ├── model2.py
│       └── model3.py
└── run_grid_search.py  # Grid search for hyperparameter tuning
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

## Colab
Run the following code in a Google Colab notebook to set up the environment and execute the grid search for model training:
```python
# Clone the repository
!git clone https://github.com/aaronbernal28/seismic-signals-tda.git

# Change to the repository directory
%cd seismic-signals-tda

# Install dependencies (assuming there's a requirements.txt file in the repo)
!pip install -r requirements.txt

# Run the grid search script
!python run_grid_search.py
```

## References

Course materials: https://mate.dm.uba.ar/~gminian/materias/tda2025/tda.html
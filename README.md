# ST-SNN: Spatio-Temporal Sheaf Neural Networks

This repository contains the implementation of **ST-SNN** (Spatio-Temporal Sheaf Neural Network), a topological extension of the ST-GCN framework for skeleton-based action recognition.

Built as a plug-in module for the [PySKL](https://github.com/kennymckormick/pyskl) ecosystem, ST-SNN replaces standard Graph Convolutional Networks (GCNs) with Sheaf Neural Networks (SheafNN). This approach leverages Sheaf Laplacians to effectively model heterophilic interactions in human skeleton graphs, preventing over-smoothing and capturing complex joint correlations.

## Performance

The architecture has been evaluated against strong baselines on standard action recognition benchmarks:

| Model | Spatial Module | Temporal Module | Accuracy |
|-------|----------------|-----------------|----------|
| ST-GCN (Baseline)| GCN | Standard | 81.5% |
| **ST-SNN** | SheafNN | Standard | **85.4%** |
| STGCN++ | GCN | MS-TCN | 89.4% |
| **ST-SNN+** | SheafNN | MS-TCN | **~89.0%** |

## Installation & Integration

This repository is designed as a direct overlay for PySKL. It contains only the novel topological modules, configuration files, and the pre-modified `__init__.py` files required for MMCV registry integration.

### 1. Prerequisites
Ensure you have a working installation of [PySKL](https://github.com/kennymckormick/pyskl) and its core dependencies.

### 2. Plug-in Installation
Simply copy and paste the contents of this repository directly into your local PySKL root directory. 

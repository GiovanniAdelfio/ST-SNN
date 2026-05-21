# ST-SNN: Spatio-Temporal Sheaf Neural Networks

This repository contains the implementation of **ST-SNN** (Spatio-Temporal Sheaf Neural Network), a topological extension of the ST-GCN framework for skeleton-based action recognition.

Built as a plug-in module for the [PySKL](https://github.com/kennymckormick/pyskl) ecosystem, ST-SNN replaces standard Graph Convolutional Networks (GCNs) with Sheaf Neural Networks (SheafNN). This approach leverages Sheaf Laplacians to effectively model heterophilic interactions in human skeleton graphs, preventing over-smoothing and capturing complex joint correlations.

## Performance

The architecture has been evaluated against strong baselines on standard action recognition benchmarks:

| Model | Spatial Module | Temporal Module | Accuracy |
|-------|----------------|-----------------|----------|
| ST-GCN (Baseline)| GCN | Standard | 81.5% |
| **ST-SNN** | SheafNN | Standard | ** 85.4%** |
| STGCN++ | GCN | MS-TCN | 89.4% |
| **ST-SNN+** | SheafNN | MS-TCN | **~89.0%** |

## Architectural Highlights

* **Orthogonal Restriction Maps via Lie Algebra:** Restriction maps $SO(d)$ are generated using the matrix exponential of learnable skew-symmetric matrices. This mapping requires only 6 independent parameters per edge for a $4 \times 4$ stalk, guaranteeing strict mathematical orthogonality while preventing over-parametrization.
* **Tensor Core Optimization:** The dynamic Sheaf Laplacian routing is structured using hardware-aware memory contiguous tensors. The `einsum` operations have been explicitly formulated to trigger cuBLAS Batched Matrix Multiplications (`bmm`) on CUDA Tensor Cores, eliminating VRAM memory thrashing and stabilizing training speeds.
* **Gradient Accumulation Integration:** Engineered to support large-capacity multi-scale temporal modules (MS-TCN) without exceeding physical VRAM limits, ensuring stable convergence through synchronized gradient accumulation steps.

## 🛠️ Installation & Integration

This repository is designed as a direct overlay for PySKL. It contains only the novel topological modules, configuration files, and the pre-modified `__init__.py` files required for MMCV registry integration.

### 1. Prerequisites
Ensure you have a working installation of [PySKL](https://github.com/kennymckormick/pyskl) and its core dependencies.

### 2. Plug-in Installation
Simply copy and paste the contents of this repository directly into your local PySKL root directory. 

```bash
# Example assuming both directories are side-by-side
cp -r ST-SNN/* pyskl/

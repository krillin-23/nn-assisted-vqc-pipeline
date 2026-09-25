# Neural Network–Assisted Variational Quantum Circuit Pipeline

A hybrid quantum-classical software pipeline for mitigating barren plateaus in Variational Quantum Circuits (VQCs) using neural-network-based parameter initialization.

## Overview

Variational Quantum Algorithms (VQAs) are promising approaches for near-term quantum computing, but their optimization can be hindered by the **barren plateau** problem, where the variance of cost-function gradients can decrease rapidly with increasing circuit depth.

This project investigates a neural-network-based initialization strategy in which a lightweight two-layer feedforward neural network generates VQC parameters that are biased toward useful regions of the optimization landscape. The approach is studied using a **6-qubit Variational Quantum Eigensolver (VQE)** for the **Transverse-Field Ising Model (TFIM)**. :contentReference[oaicite:0]{index=0}

The project consists of a training/benchmarking module and a production pipeline for using the trained models in multiple downstream tasks. :contentReference[oaicite:1]{index=1}

## Features

- Neural-network-based VQC parameter initialization
- Gradient-variance benchmarking against random initialization
- NN-assisted VQE with Adam fine-tuning
- Hamiltonian transfer benchmarking across different `(J, h)` configurations
- On-demand VQC parameter generation and ranking
- Export of trained models and VQC parameters in multiple formats
- Command-line interface for selecting individual pipelines :contentReference[oaicite:2]{index=2}

## Model Architecture

The neural network used for parameter generation is a lightweight two-layer feedforward network.

The VQC uses a hardware-efficient ansatz consisting of:

- `Ry` and `Rz` rotations on each qubit
- Linear CNOT entangling layers
- `2 × n × L` trainable parameters for `n` qubits and `L` circuit layers :contentReference[oaicite:3]{index=3}

For this project:

- Number of qubits: `6`
- Circuit depths studied: `4` and `10`
- Hamiltonian: Transverse-Field Ising Model
- Random seed: `42`

## Project Structure

```text
nn-assisted-vqc-pipeline/
│
├── src/
│   ├── Training.py
│   └── vqc_pipeline.py
│
├── trained_nn_model_depth_4.npz
├── trained_nn_model_depth_10.npz
│
├── results/
│   ├── benchmark/
│   │   ├── barren_plateau_results.png
│   │   ├── benchmark_heatmap.png
│   │   └── benchmark_results.json
│   │
│   ├── vqe/
│   │   ├── depth4/
│   │   └── depth10/
│   │
│   ├── parameter_generation/
│   │   ├── depth4/
│   │   └── depth10/
│   │
│   └── model_exports/
│       ├── depth4/
│       └── depth10/
│
├── report/
│   └── MLP_Project_Final.pdf
│
├── requirements.txt
├── LICENSE
└── .gitignore

# OpenRAN-AI-Scheduler
# AI-Driven Dynamic Spectrum Allocation for Open RAN

A reproducible simulation project demonstrating how AI/ML models can optimize spectrum allocation in an Open RAN environment using public datasets.

## Overview

This project leverages machine learning to dynamically allocate spectrum resources for cellular networks, maximizing network throughput and fairness, and utilizes only publicly available datasets and open-source tools.

## Features

- End-to-end simulation of cellular RAN with AI-based dynamic spectrum allocation
- Baseline algorithms (e.g., round-robin) for comparison
- Uses public datasets (OpenRAN Gym, Kaggle, Colosseum)
- Jupyter Notebooks for data analysis, model training, and results visualization
- Modular and extensible codebase

## Project Structure

```
├── data/            # Datasets (links or scripts to download included)
├── notebooks/       # Jupyter exploration, analysis, and demo scripts
├── src/             # Core Python modules (preprocessing, models, simulation)
├── results/         # Output figures and analysis reports
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-dynamic-spectrum-openran.git
cd ai-dynamic-spectrum-openran
```

### 2. Setup Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Download Public Datasets

- [OpenRAN Gym Datasets](https://openrangym.com/datasets)
- [Kaggle Cellular Network Analysis Data](https://www.kaggle.com/datasets/vishnuvarthanraja/cellular-network-analysis)
- [Colosseum O-RAN COMMAG Dataset](https://colosseum.net/datasets/commag)

*Place downloaded datasets in the `data/` directory as described in the data loading scripts.*

### 4. Run the Simulation

Open and execute the main notebook:

```bash
jupyter notebook notebooks/main_simulation.ipynb
```

This notebook covers:
- Data exploration and preprocessing
- Baseline and AI model definition/training
- Simulation loop and evaluation
- Visualizations and metric comparisons

## Project Pipeline

1. **Data Ingestion:** Load and preprocess RAN traffic, user, and cell data.
2. **Feature Engineering:** Construct features (user load, channel quality, cell assignment, etc.).
3. **Baseline Allocation:** Implement a static method for spectrum assignment.
4. **ML Modeling:** Train an ML model (e.g., Random Forest, neural net) to predict allocations dynamically.
5. **Simulation:** Apply the AI policy in a loop; collect KPIs (throughput, fairness).
6. **Evaluation:** Compare AI-driven and baseline spectrum strategies in diverse load scenarios.
7. **Reporting:** Generate visualizations and an analysis report.

## Results

- Plots and tables comparing network throughput, spectrum efficiency, and fairness between AI and baseline methods
- Example outputs and benchmarking metrics
- Performance analysis and limitations

## Requirements

- Python 3.7+
- scikit-learn
- TensorFlow or PyTorch
- Pandas, Numpy, Matplotlib, Seaborn
- Jupyter Notebook

Install requirements using:

```bash
pip install -r requirements.txt
```

## Limitations

- Simulated environment—physical-layer radio effects are abstracted
- Results depend on the quality/diversity of public datasets
- For production-grade evaluation, integration with real Open RAN stacks (e.g., srsRAN) is recommended

## References

- OpenRAN Gym documentation
- Recent publications on AI/ML for RAN optimization
- Kaggle and Colosseum RAN datasets

## License

MIT License – see `LICENSE` file for details.

---

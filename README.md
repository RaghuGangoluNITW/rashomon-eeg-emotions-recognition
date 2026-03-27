# An Interpretable Deep Learning Model for EEG Emotion Recognition Using the Rashomon Effect

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.bspc.2026.110140-blue.svg)](https://doi.org/10.1016/j.bspc.2026.110140)

📄 **Published Paper**: https://doi.org/10.1016/j.bspc.2026.110140

**"An Interpretable Deep Learning Model for EEG Emotion Recognition Using the Rashomon Effect"** 

---

## Overview

This work introduces a **Rashomon-set-aware Graph Convolutional Network (GCN)** framework for EEG-based emotion recognition. Rather than returning a single "best" model, we identify the full set of near-optimal models (the Rashomon set) and leverage their disagreement quantified via the **Predictive Diversity Index (PDI)** as a rich interpretability signal.

Key contributions:
- **Multi-graph topology**: 6 EEG connectivity graphs (PLV, Pearson, coherence, MI, PLI, AEC) fused via learnable weights
- **Rashomon set construction**: ε-accuracy threshold over LOSO cross-validation identifies 17-31% (DEAP) and 17-23% (DREAMER) of evaluated pipelines as near-optimal
- **PDI interpretability**: Regions of high predictive diversity correspond to emotionally ambiguous or noisy brain states, validated with SHAP and GNNExplainer
- **Cross-dataset validation**: Evaluated on DEAP (32 subjects, 40 trials) and DREAMER (23 subjects, 18 clips) with LOSO protocol

---

## Datasets

### DEAP
Koelstra et al. (2012). 32-channel EEG, 32 subjects, Valence/Arousal binarized at median.
- Download: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
- Place preprocessed `.pkl` file at `data/deap/data_preprocessed_python/`

### DREAMER
Katsigiannis & Ramzan (2018). 14-channel EEG, 23 subjects, Valence/Arousal/Dominance binarized at median.
- Download: https://zenodo.org/record/546113
- Place `DREAMER.mat` at `data/dreamer/`

---

## Installation

```bash
git clone https://github.com/RaghuGangoluNITW/rashomon-eeg-emotions-recognition.git
cd rashomon-eeg-emotions-recognition
pip install -r requirements.txt
```

**Key dependencies:** PyTorch ≥ 1.12, PyTorch Geometric, NumPy, SciPy, scikit-learn, SHAP, MNE, PyWavelets, Matplotlib, Plotly.

---

## Quick Start

### DEAP Full LOSO Rashomon pipeline
```bash
python scripts/run_full_rashomon_loso.py \
    --data_path data/deap/data_preprocessed_python/ \
    --output_dir results/deap_loso/ \
    --label valence
```

### DREAMER Full LOSO pipeline
```bash
python scripts/run_full_loso_dreamer.py \
    --data_path data/dreamer/DREAMER.mat \
    --output_dir results/dreamer_loso/ \
    --label valence
```

### Generate paper figures
```bash
python scripts/generate_final_paper_figures.py --results_dir results/
```

### Generate interactive 3D SHAP visualizations
```bash
python scripts/generate_3d_visualizations.py --output_dir figures/interactive/
```

---

## Repository Structure

```
GitHub_Release/
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
│
├── rashomon_emotion/                 # Core Python module
│   ├── model.py                      # 3-layer GCN with learnable fusion
│   ├── features.py                   # EEG feature extractors (Hjorth, spectral,
│   │                                 #   wavelet, Lorentzian, differential entropy,
│   │                                 #   fractal dim, relative bandpower)
│   ├── graph_utils.py                # Graph builders (PLV, Pearson, coherence,
│   │                                 #   MI, PLI, AEC)
│   ├── rashomon_runner.py            # LOSO loop + Rashomon set construction
│   ├── preprocessing.py             # EEG preprocessing, T=256 standardization,
│   │                                 #   VAD binarization
│   ├── interpretability.py          # SHAP (DeepExplainer), GNNExplainer, PDI
│   ├── config.py                     # Hyperparameters
│   ├── utils.py                      # Shared utilities
│   └── __init__.py
│
├── scripts/                          # Pipeline and analysis scripts
│   ├── train.py                      # Single-model training
│   ├── evaluate.py                   # Model evaluation
│   ├── run_full_rashomon_loso.py     # DEAP: 48-pipeline LOSO Rashomon
│   ├── run_dreamer_full_pipeline.py  # DREAMER: full pipeline
│   ├── run_dreamer_loso.py           # DREAMER: LOSO evaluation
│   ├── run_full_loso_dreamer.py      # DREAMER: full LOSO with Rashomon
│   ├── run_fusion_ablation_aligned.py # Fusion layer ablation study
│   ├── compute_pdi.py                # PDI computation
│   ├── generate_paper_figures.py     # Core paper figure generation
│   ├── generate_final_paper_figures.py # All final paper figures
│   ├── generate_critical_reviewer_figures.py # Per-reviewer figures
│   ├── generate_3d_visualizations.py # Interactive HTML 3D plots
│   ├── compare_deap_dreamer.py       # Cross-dataset comparison
│   ├── aggregate_deap_results.py     # DEAP results aggregation
│   ├── generate_dreamer_summaries.py # DREAMER PDI/SHAP summaries
│   ├── profile_runtime.py            # Computational complexity profiling
│   ├── overfitting_checks.py         # Training/test gap analysis
│   └── ...                           # Additional diagnostic scripts
│
├── tests/                            # Unit tests
│   ├── test_features.py
│   ├── test_model.py
│   ├── test_preprocesing.py
│   └── test_utils.py
│
├── results/
│   ├── deap/
│   │   ├── paper_ready_results.csv   # 145-row per-subject results table
│   │   ├── rashomon_set_summary.csv  # Rashomon set membership summary
│   │   └── shap_lobe_summary.csv     # SHAP importance by brain lobe
│   ├── dreamer_loso/
│   │   └── loso_summary.json         # DREAMER 23-subject LOSO summary
│   ├── dreamer_rashomon/
│   │   └── rashomon_set_aggregated.json  # DREAMER Rashomon aggregated stats
│   └── ablation/
│       ├── fusion_ablation_results.json  # Fusion layer ablation results
│       ├── fusion_ablation_summary.csv   # Ablation summary table
│       ├── fusion_ablation_detailed.csv  # Per-configuration detailed results
│       ├── ablation_summary_DEAP.json    # DEAP ablation summary
│       └── ablation_aggregated.csv       # Aggregated ablation statistics
│
├── figures/
│   ├── paper_ready/                  # 11 publication-quality figures (PNG)
│   │   ├── training_curves_final.png
│   │   ├── confusion_matrix_deap.png
│   │   ├── confusion_matrix_dreamer.png
│   │   ├── f1_boxplot_top5.png
│   │   ├── rashomon_pdi_heatmap.png
│   │   ├── dreamer_pdi_heatmap.png
│   │   ├── pdi_accuracy_scatter.png
│   │   ├── rashomon_set_analysis.png
│   │   ├── shap_lobe_comparison.png
│   │   ├── cross_dataset_pdi.png
│   │   └── ...
│   ├── dreamer/                      # 10 DREAMER-specific analysis figures (PNG)
│   │   ├── dreamer_per_subject_accuracy.png
│   │   ├── dreamer_accuracy_vs_pdi_scatter.png
│   │   ├── dreamer_auc_top5_boxplots.png
│   │   ├── dreamer_f1_top5_boxplots.png
│   │   └── ...
│   └── interactive/                  # 11 interactive 3D HTML visualizations
│       ├── index.html                # Navigation page open this first
│       ├── shap_3d_valence.html
│       ├── pdi_3d_surface.html
│       ├── rashomon_set_3d.html
│       └── ...
│
└── logs/
    ├── dreamer_loso/                 # Per-subject LOSO logs (23 JSONs + summary)
    │   ├── loso_subject_01.json
    │   ├── ...
    │   └── loso_summary.json
    └── deap_ablation/                # DEAP training history per subject/config
        ├── history_sub1_wavelet_hid64.json
        └── ...
```

---

## Model Architecture

The core model (`rashomon_emotion/model.py`) is a 3-layer GCN:

1. **Multi-graph input**: 6 connectivity graphs built per EEG trial
2. **GCN layers**: 3 × GraphConv with ReLU, dropout (p=0.5), batch normalization
   - Hidden sizes: 64 → 128 → 64
3. **Learnable fusion**: Softmax-weighted combination of all 6 graph outputs
4. **Output head**: Sigmoid for binary Valence and Arousal classification

**Hyperparameters** (`rashomon_emotion/config.py`):
- Optimizer: Adam, lr=0.001, weight_decay=1e-4
- Epochs: 100, batch size: 32
- ε threshold for Rashomon set: 0.95 × max accuracy

---

## Rashomon Set Construction

Given N evaluated pipelines (feature set × graph topology combinations), the Rashomon set is defined as:

```
R(ε) = { f ∈ F : Accuracy(f) ≥ (1 - ε) × Accuracy(f*) }
```

where `f*` is the best-performing pipeline. `ε` is set so that 17-31% of pipelines qualify on DEAP and 17-23% on DREAMER. See `rashomon_emotion/rashomon_runner.py` for implementation.

### PDI (Predictive Diversity Index)

For each EEG trial `x`, PDI measures the fraction of Rashomon-set models that disagree with the majority vote:

```
PDI(x) = (1 / |R|) × Σ 1[f(x) ≠ majority_vote(x)]
```

High PDI = emotionally ambiguous input. Computed in `scripts/compute_pdi.py`.

---

## Reproducing Paper Results

### Table 1 DEAP LOSO accuracies
```bash
python scripts/run_full_rashomon_loso.py --data_path data/deap/ --label valence
python scripts/run_full_rashomon_loso.py --data_path data/deap/ --label arousal
python scripts/aggregate_deap_results.py --results_dir results/deap_loso/
```

### Table 2 DREAMER LOSO accuracies
```bash
python scripts/run_full_loso_dreamer.py --data_path data/dreamer/DREAMER.mat
```

### Table 3 Fusion ablation
```bash
python scripts/run_fusion_ablation_aligned.py
```

### Figure generation (all paper figures)
```bash
python scripts/generate_final_paper_figures.py
```

### Cross-dataset comparison figure
```bash
python scripts/compare_deap_dreamer.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Key Results

### DEAP (32 subjects, 40 trials each from `results/deap/paper_ready_results.csv`)

| Label     | Accuracy | F1     | Test samples |
|-----------|----------|--------|--------------|
| Arousal   | 80.0%    | 88.9%  | 120          |
| Valence   | 74.2-77.5% | 84.6-87.0% | 120    |
| Dominance | 85.8%    | 92.4%  | 120          |

Metrics computed over a 3-subject LOSO test fold × 40 trials (120 samples) across 48 pipelines (8 feature sets × 6 graph topologies). Results are derived from prediction pickle files in `Rahomon_DEAP/Pickles/`.

### DREAMER (23 subjects, 18 clips each from `results/dreamer_loso/loso_summary.json`)

| Metric           | Value              |
|------------------|--------------------|
| Mean accuracy    | 60.6% (±10.4%)     |
| Median accuracy  | 66.7%              |
| Range            | 33.3%-77.8%        |
| Feature set      | Wavelet            |
| Graph topologies | PLV, Coherence, Correlation, MI, AEC |
| Subjects         | 23 (LOSO)          |

DREAMER LOSO logs available per-subject in `logs/dreamer_loso/`.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions about the code or experiments, please open a GitHub issue.

## 📄 Paper (Published)

**An Interpretable Deep Learning Model for EEG Emotion Recognition Using the Rashomon Effect**

Published in *Biomedical Signal Processing and Control (BSPC), 2026*.

🔗 DOI: https://doi.org/10.1016/j.bspc.2026.110140


### 📚 Citation

If you use this work, please cite:

```bibtex
@article{gangolu2026rashomon,
  title={An Interpretable Deep Learning Model for EEG Emotion Recognition Using the Rashomon Effect},
  author={Gangolu, Raghu and others},
  journal={Biomedical Signal Processing and Control},
  year={2026},
  doi={10.1016/j.bspc.2026.110140}
}


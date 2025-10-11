
# Rashomon-Effect-Based Interpretable EEG Emotion Recognition: Supplementary Material

## Abstract
This repository provides a modular and interpretable deep learning framework for **EEG-based emotion recognition** inspired by the *Rashomon Effect*.  
The framework aims to achieve **high classification performance** while exploring **multiple plausible models** and providing insights into model decisions using **SHAP values**, **attention mechanisms**, and **graph-based feature analysis**.  
We demonstrate the methodology on the **DEAP dataset**, employing **wavelet-based preprocessing**, **handcrafted feature extraction**, and **graph-based deep learning**.



## Index Terms
EEG, Emotion Recognition, Rashomon Effect, Interpretability, Deep Learning, SHAP, Graph Neural Networks, DEAP Dataset.



## 1. Introduction
Emotion recognition from EEG signals is critical for affective computing, human-computer interaction, and mental health monitoring. Traditional approaches focus solely on maximizing classification accuracy, often neglecting interpretability and model diversity.  
Inspired by the *Rashomon Effect*—the concept that multiple distinct models can explain the same dataset equally well—this work emphasizes **understanding model decisions** while maintaining competitive performance.  
Interpretability is achieved through **SHAP-based feature attribution**, **attention map visualization**, and **graph representations of EEG channels**.



## 2. Repository Overview

### 2.1 Structure
```
rashomon-emotion/
├── data/                     # External EEG dataset (DEAP)
├── rashomon_emotion/         # Core modular source code
├── scripts/                  # Training, evaluation, visualization
├── tests/                    # Unit and integration tests
├── requirements.txt          # Dependencies
└── README.md                 # Supplementary material
```

### 2.2 Installation
```bash
git clone https://github.com/RaghuGangoluNITW/rashomon-eeg-emotions-recognition.git
cd rashomon-eeg-emotions-recognition
pip install -r requirements.txt
```



## 3. Dataset
The **DEAP dataset** provides EEG, peripheral physiological signals, and video recordings for emotion analysis.  
- **Modalities**: 32-channel EEG, peripheral physiological signals, and facial video.  
- **Labels**: Valence, arousal, dominance, and liking ratings.  
- **Access**: Must be requested from the dataset authors.  
Place downloaded data in the `data/` directory.



## 4. Methodology Summary

### 4.1 Preprocessing
- Wavelet-based denoising to remove artifacts.  
- Handcrafted feature extraction (power spectral density, bandpower features).  

### 4.2 Model Architecture
- **Graph Neural Networks (GNNs)** via PyTorch Geometric to capture inter-channel relationships.  
- Multiple plausible models explored to implement the **Rashomon Effect**.

### 4.3 Dimensionality Analysis
- **UMAP** is used for embedding EEG representations and clustering similar trials.  

### 4.4 Interpretability
- **SHAP values** quantify feature contributions.  
- **Attention maps** highlight critical channels and time windows.  
- **Graph visualizations** reveal spatial and functional relationships between electrodes.



## 5. Experimental Setup

### 5.1 Training
```bash
python scripts/train.py
```

### 5.2 Evaluation
```bash
python scripts/evaluate.py
```

### 5.3 Visualization
```bash
python scripts/visualize.py
```
This generates embedding plots, attention maps, and SHAP-based feature attributions.



## 6. Rashomon Effect in EEG Emotion Recognition
> The Rashomon Effect describes scenarios where **multiple distinct models explain the same dataset with comparable accuracy**.  
In this work, we demonstrate that **diverse models** can achieve similar performance on DEAP EEG data while providing **complementary interpretability insights**.



## 7. Reproducibility Statement
- **Dependencies**: Listed in `requirements.txt`.  
- **Data**: DEAP dataset, external download required.  
- **Scripts**: All code for training, evaluation, and visualization is modularized under `scripts/`.  
- **Random Seeds**: Settable in configuration files for reproducibility.  
- **Hardware**: Compatible with GPU and CPU, PyTorch 2.x recommended.



## 8. Citation
```
Raghu Gangolu, "Rashomon-Effect-Based Interpretable EEG Emotion Recognition," GitHub Repository, 2025.
```



## 9. License
This repository is released under the **MIT License**.

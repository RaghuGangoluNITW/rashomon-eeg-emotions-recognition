# 🧠 Rashomon-Effect-Based Interpretable EEG Emotion Recognition

This repository contains a modular, interpretable deep learning framework to classify emotions from EEG signals using the **Rashomon Effect**. The aim is not only to achieve high classification accuracy but also to explore **multiple plausible models** and provide insights into **why** and **how** decisions are made using SHAP, attention maps, and graph structures.

## 🚀 Project Highlights

- EEG emotion recognition using the DEAP dataset.
- Wavelet-based preprocessing and handcrafted feature extraction.
- Deep learning with PyTorch and Torch-Geometric.
- Dimensionality reduction and clustering using UMAP.
- Interpretability using SHAP and visual analysis.
- Rashomon effect modeling: multiple models explaining same data.

## 📁 Project Structure

```
rashomon-emotion/
├── data/                     # EEG dataset (external)
├── rashomon_emotion/        # Modular source code
├── scripts/                 # Train, evaluate, visualize scripts
├── tests/                   # Unit tests
├── requirements.txt
└── README.md
```

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/rashomon-emotion.git
cd rashomon-emotion
pip install -r requirements.txt
```

## 📊 Dataset

This project uses the [DEAP dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/), a multimodal dataset for emotion analysis.  
> 🔐 You'll need to request access to download it.

Place the downloaded data in the `data/` folder.

## ⚙️ Usage

### Train model
```bash
python scripts/train.py
```

### Evaluate model
```bash
python scripts/evaluate.py
```

### Visualize embeddings and SHAP values
```bash
python scripts/visualize.py
```

## 📌 Rashomon Effect?

> The **Rashomon Effect** is the phenomenon where multiple models can explain the same data equally well.  
> This project explores multiple interpretable models — encouraging openness to diverse valid solutions in AI.

## 🧠 Citation

If you use this work in your research, please cite it as:

```text
[Author's name]. Rashomon-Effect-Based Interpretable EEG Emotion Recognition. GitHub, 2025.
```

## 📜 License

This project is open-sourced under the MIT License.

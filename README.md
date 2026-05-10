# FENE: Species-Conditioned Expert Network with Environmental Fields

FENE is a deep learning framework for **species distribution modeling (SDM)** that combines species-conditioned environmental learning, continuous spatial field representation, key-feature transformation, and uncertainty-aware prediction within a unified Mixture-of-Experts architecture.

The framework is designed to model complex ecological relationships between environmental variables, species identity, and geospatial distributions while providing calibrated uncertainty estimates and explainability.

---

## Key Features

### Species-Conditioned Feature Learning
- FiLM-based species-conditioned environmental feature extraction
- Shared environmental encoder with species-specific modulation
- Adaptive representation learning across multiple species

### Continuous Spatial Field Modeling
- Fourier positional encoding for longitude/latitude representation
- SIREN-based implicit neural spatial fields
- Smooth geospatial representation learning

### Key-Feature Transformation
- Dedicated nonlinear branch for ecologically important predictors
- Automatic key-feature selection support
- Enhanced modeling of dominant environmental drivers

### Fusion Mixture-of-Experts (F-MoE)
- Multi-expert probabilistic prediction head
- Adaptive expert weighting via learned gating network
- Improved modeling of heterogeneous ecological patterns

### Uncertainty Quantification
- Predictive uncertainty estimation
- Confidence interval generation
- Spatial uncertainty visualization

### Explainability and Baselines
- SHAP-based feature attribution analysis
- Comparison with traditional machine learning baselines:
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
  - Decision Tree

---

# Repository Structure

```text
FENE/
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── fene.py
│   ├── scfe.py
│   ├── siren.py
│   ├── key_feature.py
│   ├── moe.py
│   └── losses.py
│
├── utils/
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── visualization.py
│   └── fourier_features.py
│
├── baselines/
│   ├── random_forest.py
│   ├── xgboost_model.py
│   ├── svm_model.py
│   └── decision_tree.py
│
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   ├── predictions/
│   └── shap/
│
├── train.py
├── evaluate.py
├── inference.py
├── shap_analysis.py
├── uncertainty_maps.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Installation

```bash
git clone https://github.com/ntamzi/FENE.git
cd FENE

python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

---

# Dataset Format

Place your processed dataset at:

```text
data/processed/fish_dataset.csv
```

Expected CSV format:

```text
longitude,
latitude,
species,
occurrence_probability,
environmental_feature_1,
environmental_feature_2,
...
```

---

# Training FENE

```bash
python train.py
```

Training outputs include:

```text
outputs/checkpoints/best_model.pth
outputs/logs/training_log.csv
outputs/figures/prediction_scatter.png
```

---

# Model Evaluation

```bash
python evaluate.py
```

Evaluation metrics include:

- RMSE
- MAE
- R² Score
- Predictive uncertainty statistics

---

# Inference

```bash
python inference.py \
    --input data/processed/new_samples.csv \
    --output outputs/predictions/inference_predictions.csv
```

---

# Uncertainty Visualization

```bash
python uncertainty_maps.py
```

---

# SHAP Explainability Analysis

```bash
python shap_analysis.py
```

---

# Baseline Models

```bash
python baselines/random_forest.py
python baselines/xgboost_model.py
python baselines/svm_model.py
python baselines/decision_tree.py
```

---

# Citation

```bibtex
@article{Tamzi2026FENE,
  title   = {FENE: Fish-conditioned Expert Network Using Deep Learning for Multi-species Marine Distribution Modeling},
  author  = {Tamzi, Nafisa Nawar and Rahman, Md Motiur and Bhatt, Smriti and Faezipour, Miad},
  journal = {Computers and Electronics in Agriculture},
  volume  = {249},
  pages   = {111835},
  year    = {2026},
  issn    = {0168-1699},
  doi     = {10.1016/j.compag.2026.111835},
  url     = {https://www.sciencedirect.com/science/article/pii/S0168169926004308},
}
```

---

# License

MIT License

# FENE / FENE

Species-Conditioned Expert Network with Environmental Fields for Species Distribution Modeling.

This codebase implements the architecture described in the uploaded paper: species-conditioned FiLM feature extraction, Fourier/SIREN spatial encoding, key-feature transformation, mixture-of-experts prediction, uncertainty estimation, SHAP analysis, and baseline models.

## Repository Structure

```text
FENE/
├── config/config.yaml
├── models/
├── utils/
├── baselines/
├── train.py
├── evaluate.py
├── inference.py
├── shap_analysis.py
├── uncertainty_maps.py
└── outputs/
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/FENE.git
cd FENE
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Format

Place your processed dataset at:

```text
data/processed/fish_dataset.csv
```

Expected columns:

```text
longitude, latitude, species, occurrence_probability, environmental_feature_1, environmental_feature_2, ...
```

You can edit column names in `config/config.yaml`.

## Train FENE

```bash
python train.py
```

The best model is saved to:

```text
outputs/checkpoints/best_model.pth
```

## Evaluate

```bash
python evaluate.py
```

## Inference

```bash
python inference.py --input data/processed/new_samples.csv --output outputs/logs/inference_predictions.csv
```

## Uncertainty Map

```bash
python uncertainty_maps.py
```

## SHAP Analysis

```bash
python shap_analysis.py
```

## Run Baselines

```bash
python baselines/random_forest.py
python baselines/xgboost_model.py
python baselines/svm_model.py
python baselines/decision_tree.py
```

## Citation

```bibtex
@article{tamzi2026FENE,
  title={FENE: Species-Conditioned Expert Network with Environmental Fields for Species Distribution Modeling},
  author={Tamzi, Nafisa Nawar and Rahman, Md Motiur and Bhatt, Smriti and Faezipour, Miad},
  year={2026}
}
```

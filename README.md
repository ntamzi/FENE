# FENE: Fish-conditioned Expert Network with Environmental Fields for Distribution Modeling

Official implementation of **FENE**, an end-to-end deep learning framework for multi-species marine species distribution modeling (SDM) with:
- species-conditioned environmental responses (FiLM + learned species embeddings),
- continuous spatial field modeling (Fourier positional encoding + SIREN),
- key-feature transformation for dominant environmental drivers,
- probabilistic Mixture-of-Experts head for uncertainty estimation,
- explainability via SHAP.

![Status](https://img.shields.io/badge/Code%20Release-In%20Progress-orange) The full implementation of S-CENE will be publicly released on GitHub upon acceptance of the manuscript.

## Paper
**FENE: Fish-conditioned Expert Network with Environmental Fields for Distribution Modeling**  
Nafisa Nawar Tamzi, Md Motiur Rahman, Smriti Bhatt, Miad Faezipour  


## Key Results (from the paper)
- Best overall performance vs. DT / RF / SVM / XGBoost on eight Mediterranean pelagic fish species.
- Robust improvements under aggressive feature reduction.
- Spatially explicit uncertainty maps for monitoring prioritization.
- Species-specific drivers and hotspots via SHAP.

## Data Availability

This work uses a public dataset (8 Mediterranean Pelagic Commercial Fish Species) by Effrosynidis et al. (2020) for Mediterranean pelagic fish species distribution modeling. The dataset can be downloaded from: https://data.mendeley.com/datasets/rtphmvz5fj/1

---

## Citation

```yaml
@misc{tamzi2026scene,
  title   = {FENE: Fish-conditioned Expert Network with Environmental Fields for Distribution Modeling},
  author  = {Tamzi, Nafisa Nawar and Rahman, Md Motiur and Bhatt, Smriti and Faezipour, Miad},
  year    = {2026},
  note    = {Manuscript submitted for publication},
  howpublished = {\url{https://github.com/ntamzi/FENE}}
}


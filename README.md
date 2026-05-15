# AI for Chemistry: Glycan Biomarker Discovery for Lung Cancer Detection

&nbsp;

A machine learning pipeline for discovery and molecular interpretation of glycan biomarkers in LC-MS data from a multi-cohort lung cancer study. Data was provided by [Isospec](https://isospec.io/), a biotech startup specializing in glycan characterization technology.

---

## Project Overview

Glycans are complex sugar molecules attached to proteins and lipids on cell surfaces and in blood. They undergo significant structural and abundance changes during disease progression, making them attractive candidates for early diagnostic biomarkers. This project applies end-to-end data science and machine learning to a glycomic LC-MS dataset to:

1. **Discover statistically significant and discriminative glycan biomarkers** from peak area intensities across lung cancer, benign disease, and healthy control cohorts.
2. **Interpret discovered biomarkers at the molecular level** by placing them in the context of known glycan structures, disease associations, and protein interactions using structure-based ML and embedding analysis.

---

## Dataset

**Source:** Isospec — LC-MS glycomics experiment on a lung cancer cohort study

| Property | Value |
|---|---|
| Cohorts | Lung cancer (French, n=26), Benign disease (LMU, n=26), Healthy control (Dunn, n=27) |
| Raw features | 252 LC-MS chromatographic peaks |
| Features after QC | 194 (77% retention) |
| Identified glycans | 5 sequences characterized via CIRIS technology |
| Reference database | ~50,500 glycan sequences (glycowork `df_glycan`) |
| Protein-binding data | >790,000 interactions from >2,000 proteins (`glycan_binding`) |

Each feature is described by its retention time (RT), mass-to-charge ratio (m/z), and peak area intensity. Sample classes include biological samples, pooled quality controls (QC), blanks, and system suitability standards.

---

## Repository Structure

```
AI4chem/
├── Notebooks/
│   ├── 00_environment_check.ipynb       Environment and dependency verification
│   ├── 01_data_loading.ipynb            Raw data ingestion and alignment
│   ├── 02_eda.ipynb                     Exploratory data analysis
│   ├── 03_preprocessing.ipynb           QC filtering and normalization
│   ├── 04_statistical_analysis.ipynb    Univariate tests, PLS-DA, volcano plots
│   ├── 05_ml_classification.ipynb       ML classification, SHAP, nested CV
│   ├── 06_glycan_enrichment.ipynb       Disease/tissue/protein enrichment
│   ├── 07_glycan_embedding.ipynb        Structure-based ML and UMAP embedding
│   └── 08_summary_figures.ipynb         Publication-ready figure panel
├── files/
│   ├── data/
│   │   ├── input/                       Raw LC-MS CSV files
│   │   ├── processed/                   Intermediate pickle files
│   │   └── glycan_embedding/            Glycowork reference databases + identified glycans
│   └── results/                         statistical_results.pkl, model_results.pkl,
│                                        enrichment_results.pkl, embedding_results.pkl
├── figures/                             PNG figures from all analysis phases
├── requirements.txt
└── README.md
```

---

## Methods

### Phase 1 — Exploratory Data Analysis (`02_eda.ipynb`)

- Feature map visualization across m/z × retention time space
- Isomer and isotope/adduct detection
- Blank contamination assessment: 0 features flagged (glycan mass range is intrinsically clean)
- Exogenous standard recovery: all 4 spiked-in standards detected at expected m/z and RT
- Coefficient of Variation (CV) on QC samples: **78.2% of features have CV < 30%**
- D-Ratio analysis to compare technical vs. biological variation

### Phase 2 — Preprocessing (`03_preprocessing.ipynb`)

Four sequential filters applied:

| Filter | Criterion | Removed |
|---|---|---|
| Mass range | m/z > 500 Da (glycan-sized) | 4 |
| Prevalence | ≥ 70% detection in at least one biological class | 0 |
| QC CV | < 30% | 55 |
| Contamination | Blank/bio ratio ≤ 30% | 0 |

**Result: 194 features retained (77%)**

Normalization pipeline:
- Half-minimum imputation (< 1% missing data)
- Log₂ transformation
- Probabilistic Quotient Normalization (PQN) for sample loading correction
- Run-order drift flag: 30 features with monotonic decay detected (not confounded with class, Kruskal-Wallis p = 0.257)

### Phase 3 — Statistical & Discriminatory Analysis (`04_statistical_analysis.ipynb`)

- **PCA:** 46% variance explained in PC1–PC2; 7–8 components reach 80% cumulative variance
- **UMAP:** Non-linear embedding reveals well-separated disease clusters
- **Univariate testing:** Kruskal-Wallis + Benjamini-Hochberg FDR correction
  - **181 / 194 features significant** (q < 0.05)
  - Median effect size (η²) = 0.45 (large)
- **Volcano plots:** |log₂FC| > 1 AND q < 0.05 identifies high-magnitude candidates
  - FT-174: +4.4 log₂FC (21-fold up in cancer vs healthy)
  - FT-172: +3.4 log₂FC (10-fold up in cancer vs healthy)
- **PLS-DA:** Q² = 0.949 (in-sample); LOO-CV Q² = 0.956; permutation test p = 0.000

### Phase 4 — Machine Learning Classification (`05_ml_classification.ipynb`)

Models trained on 79 biological samples × 194 features:

| Model | CV Macro F1 | Test F1 | Test AUC |
|---|---|---|---|
| Logistic Regression (L2) | 1.0 | 1.0 | 1.0 |
| Random Forest | 1.0 | 1.0 | 1.0 |

Validation strategy:
- Leave-One-Out CV balanced accuracy: **1.0**
- Repeated 5-fold CV (10 repeats): **1.0 ± 0.0**
- Nested 5-fold CV (unbiased estimate): **all 5 outer folds achieve F1 = 1.0**

SHAP feature importance computed for both models. Cross-method consensus (RF SHAP, LR SHAP, statistical composite) identifies **6 overlapping top-20 features**. Unbiased nested-CV consensus panel: **FT-046, FT-049, FT-050, FT-070** (m/z ~1004–1004.7 Da, RT ~1156 s, η² = 0.97–0.98).

### Phase 5a — Glycan Enrichment (`06_glycan_enrichment.ipynb`)

Five glycans identified by CIRIS technology were annotated against the glycowork reference database:

| Glycan | Composition | DB Match | Cancer Diseases |
|---|---|---|---|
| G0F2 | dHex₂Hex₃HexNAc₄ | 73 composition matches | colorectal (1) |
| A1F1 | Neu5Ac₁Hex₄HexNAc₄dHex₁ | 53 composition matches | colorectal (6), gastric (1) |
| A2G2S1 | Neu5Ac₁Hex₅HexNAc₄ | **1 exact match** | — (1663 protein interactions) |
| A1G1S1 | Neu5Ac₁Hex₄HexNAc₄ | **1 exact match** | — (1585 protein interactions) |
| A2G2F1 | dHex₁Hex₅HexNAc₅ | **1 exact match** | — |

Top enriched disease terms across all 5 glycans: colorectal cancer (7), gastric cancer (1), lung NSCC (1), esophageal cancer (1), breast cancer (1). Dominant tissue associations: gastric mucosa, blood serum, lung epithelium.

Each glycan was converted to a directed graph representation (NetworkX) with monosaccharide nodes and glycosidic linkage edges for structure visualization.

### Phase 5b — Structure-Based Classification (`07_glycan_embedding.ipynb`)

**Goal:** Predict cancer association from glycan molecular structure alone, independent of LC-MS.

Training set: 1,500 glycans (501 cancer-associated + 999 non-cancer) from `df_glycan`. Features: 1,445 structural motif fingerprints via `glycowork.annotate_dataset`. Label: binary (any cancer/carcinoma/tumor in disease association).

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.922 | 0.825 |
| Random Forest | 0.885 | 0.788 |
| **XGBoost** | **0.921** | **0.843** |

Top structural motifs by XGBoost gain: **Neu5Ac (sialic acid, gain 0.0747)**, GlcNAc(b1-4/6)Man (0.0664), Internal LacNAc type-2 (0.0267).

Cancer probability predictions for the 5 identified biomarker glycans:

| Glycan | P(cancer) | Structural verdict |
|---|---|---|
| A1F1 | 0.528 | Cancer |
| A2G2S1 | 0.520 | Cancer |
| A1G1S1 | 0.514 | Cancer |
| A2G2F1 | 0.279 | Not cancer |
| G0F2 | 0.128 | Not cancer |

### Phase 5c — UMAP Embedding & Nearest-Neighbour Analysis (`07_glycan_embedding.ipynb`)

UMAP embedding of 1,505 glycans (1,500 training + 5 biomarkers) using Jaccard distance on structural fingerprints (n_neighbors=15, min_dist=0.1). K=10 nearest-neighbour cancer enrichment (background rate: 33%):

| Glycan | Cancer neighbours / 10 | Mean Jaccard dist |
|---|---|---|
| A1F1 | 7/10 (70%) | 0.563 |
| A1G1S1 | 7/10 (70%) | 0.594 |
| A2G2S1 | 6/10 (60%) | 0.590 |
| G0F2 | 5/10 (50%) | 0.513 |
| A2G2F1 | 4/10 (40%) | 0.632 |

---

## Key Findings

### Task 1 — Biomarker Discovery

- **Perfect three-class discrimination** achieved: both Logistic Regression and Random Forest reach F1 = 1.0 and AUC = 1.0, validated by nested cross-validation.
- **4-feature consensus biomarker panel** (FT-046, FT-049, FT-050, FT-070): unbiased across all validation strategies, explaining 97–98% of between-class variance.
- **High-magnitude outliers** FT-172 and FT-174 show 10–21× fold-increase in cancer vs healthy — strong biological signal but near the analytical limit of detection in QC.
- **181 of 194 features** are statistically significant after FDR correction, indicating broad glycome remodelling across disease classes.

### Task 2 — Molecular Interpretation

- **Three sialylated glycans** (A1F1, A2G2S1, A1G1S1) show convergent evidence across three independent lines of analysis:
  - Structural cancer probability > 0.50 (XGBoost)
  - 7/10 cancer-labelled structural neighbours in UMAP space
  - Exact database matches with documented human protein binding and tissue associations
- **Sialic acid (Neu5Ac) is the dominant cancer-predictive structural motif**, consistent with the known role of hypersialylation in cancer progression.
- **Two non-sialylated glycans** (G0F2, A2G2F1) have lower structural cancer scores and sparser nearest-neighbour cancer enrichment; their discriminatory power in the LC-MS data may reflect cancer-specific expression remodelling rather than a globally conserved cancer structural signature.

---

## Reproducibility

### Environment Setup

```bash
conda create -n ai4chem python=3.10
conda activate ai4chem
pip install -r requirements.txt
```

### Running the Analysis

Notebooks are numbered and should be run in order. Each notebook saves its outputs to `files/results/` as pickle files, which are loaded by subsequent notebooks.

```
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

### Dependencies

Core libraries used:

| Category | Libraries |
|---|---|
| Data | pandas 2.3.3, numpy 2.2.1 |
| Statistics | scipy 1.14.1, statsmodels 0.14.4 |
| Machine Learning | scikit-learn 1.6.0, xgboost 3.2.0 |
| Interpretability | shap 0.49.1 |
| Dimensionality Reduction | umap-learn 0.5.12 |
| Glycan-specific | glycowork 1.7.0, glycorender 0.1.0, networkx 3.4.2 |
| Visualization | matplotlib 3.10.0, seaborn 0.13.2 |

---

## Resources

- [LC-MS Background — PyOpenMS](https://pyopenms.readthedocs.io/en/latest/user_guide/background.html)
- [D-Ratio (technical vs biological variation)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10222478/)
- [Coefficient of Variation in LC-MS](https://pmc.ncbi.nlm.nih.gov/articles/PMC3695475/)
- [Glycan Analysis Review](https://www.mdpi.com/2218-273X/13/4/605)
- [Glycowork library](https://github.com/BojarLab/glycowork)
- [Isospec — CIRIS technology](https://isospec.io/)

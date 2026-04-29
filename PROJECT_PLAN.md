# Machine Learning-Driven Discovery of Glycan Biomarkers for Lung Cancer Detection

**Course:** AI for Chemistry  
**Subject Area:** Glycoproteomics / Computational Glycobiology  
**Dataset:** LC-MS Glycomics Cohort (Lung Cancer, Benign Disease, Healthy Controls)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background & Motivation](#2-background--motivation)
3. [Research Questions](#3-research-questions)
4. [Dataset Description](#4-dataset-description)
5. [Methodology](#5-methodology)
   - [Phase 1: Exploratory Data Analysis](#phase-1-exploratory-data-analysis)
   - [Phase 2: Data Quality Control & Filtering](#phase-2-data-quality-control--filtering)
   - [Phase 3: Statistical & Discriminatory Analysis](#phase-3-statistical--discriminatory-analysis)
   - [Phase 4: Machine Learning Classification](#phase-4-machine-learning-classification)
   - [Phase 5: Glycan Embedding & Biological Interpretation](#phase-5-glycan-embedding--biological-interpretation)
   - [Phase 6: Visualization & Summary](#phase-6-visualization--summary)
6. [Tools & Libraries](#6-tools--libraries)
7. [Expected Outcomes](#7-expected-outcomes)
8. [Project Timeline](#8-project-timeline)
9. [Documentation Plan](#9-documentation-plan)

---

## 1. Project Overview

This project applies machine learning and statistical analysis to a liquid chromatography–mass spectrometry (LC-MS) glycomics dataset from a multi-cohort lung cancer study. The dataset contains molecular expression profiles from three patient groups: lung cancer patients, individuals with benign pulmonary disease, and healthy controls. By combining rigorous data quality control, classical statistics, and modern machine learning, we aim to identify glycan features that reliably distinguish disease states — a process known as **biomarker discovery**.

The second component of the project moves beyond feature selection into **glycan embedding**: representing individual glycan molecules as points in a learned vector space that captures structural similarity, biological origin, and disease relevance. This enables richer interpretation of the discovered biomarkers and opens the door to transfer learning across glycan datasets.

Together, the two tasks demonstrate a complete AI-for-chemistry pipeline: from raw instrument data to biologically interpretable, machine-learning-validated molecular insights.

---

## 2. Background & Motivation

### What Are Glycans?

Glycans are complex carbohydrate molecules that are covalently attached to proteins and lipids on the surfaces of cells. They are among the most structurally diverse biological molecules, and their composition changes significantly in response to disease, developmental state, and environmental stress. In cancer, altered glycosylation patterns have been observed across many tumour types, making glycans valuable candidates for non-invasive biomarker discovery.

### LC-MS as a Detection Technology

Liquid chromatography–mass spectrometry (LC-MS) is the gold-standard analytical platform for untargeted glycan profiling. It separates molecules by their chromatographic retention time and identifies them by their mass-to-charge ratio (m/z). A single LC-MS experiment on a biological sample produces thousands of features, each representing a detected molecular signal at a specific (m/z, retention time) coordinate. The intensity of each feature reflects the relative abundance of that molecule in the sample.

### Why Machine Learning?

The LC-MS data matrix is high-dimensional (many more features than samples), noisy, and contains batch effects introduced by instrument variation across experimental runs. Classical statistical tests alone are insufficient to identify the most informative features, control for confounders, and build predictive models. Machine learning provides:

- **Dimensionality reduction** — to visualise and explore structure in high-dimensional data
- **Feature selection** — to rank features by discriminatory power
- **Classification** — to build predictive models that generalise to new patients
- **Interpretability** — to translate model predictions back into molecular biology

### Scientific Significance

Lung cancer is the leading cause of cancer-related mortality worldwide. Early detection dramatically improves survival rates, but reliable non-invasive biomarkers remain elusive. Identifying glycan signatures that distinguish early-stage disease from healthy states could contribute to the development of blood-based diagnostic tests.

---

## 3. Research Questions

1. **Do glycan expression profiles differ significantly between lung cancer patients, benign disease patients, and healthy controls?**
2. **Which glycan features are the strongest discriminators between disease states, and do they fall within a biologically meaningful mass range?**
3. **Can machine learning classifiers trained on glycan profiles achieve high accuracy and generalise beyond the training set?**
4. **Which molecular features drive model predictions, and are these consistent with known glycan biology?**
5. **Does a learned glycan embedding space reflect known structural, functional, and disease-related properties of glycan molecules?**

---

## 4. Dataset Description

### Cohort Design

The dataset comes from a multi-site LC-MS glycomics study with three biological classes:

| Class Label | Group | Description |
|-------------|-------|-------------|
| `French` | French cohort | Confirmed lung cancer patients (Disease) |
| `LMU` | LMU cohort | Patients with benign pulmonary disease (Benign) |
| `Dunn` | Dunn cohort | Healthy volunteers (Control) |

In addition to biological samples, the dataset contains several types of quality control samples:

| Sample Type | Purpose |
|-------------|---------|
| `QC` | Pooled quality control — injected repeatedly across batches to assess instrument stability |
| `dQC` | Diluted quality control — used to assess linearity and signal dynamic range |
| `SS` | System suitability — verifies instrument performance before each batch |
| `B` | Solvent blank — identifies background/contaminant signals (note: `B` in the data means Blank, not Benign) |

### Data Files

| File | Description |
|------|-------------|
| `data/input/data_matrix.csv` | N × M matrix of LC-MS peak intensities. Rows = samples, columns = feature IDs (e.g., FT-000, FT-001, …). Values are integrated peak areas. |
| `data/input/acquisition_list.csv` | One row per sample. Columns: sample ID, class label, batch number, run order. |
| `data/input/feature_metadata.csv` | One row per feature. Columns: feature ID, m/z (mean, min, max, std), retention time (mean, min, max, std, start, end). |
| `data/input/exogenous_standards.csv` | Reference table of spiked-in exogenous glycan standards (GU4, GU5, GU14, GU15, …) with known m/z and expected retention times. Used to verify instrument calibration. |
| `data/glycan_embedding/glycan_list.csv` | Annotated list of glycan sequences identified from the LC-MS data. Columns: IUPAC sequence notation, monosaccharide composition (dHex, Hex, HexNAc, Neu5Ac counts), tissue species, tissue sample type. |
| `data/glycan_embedding/df_glycan.pkl` | Reference database of ~50,500 unique glycan sequences with associated species, tissue, and disease labels. |
| `data/glycan_embedding/glycan_binding.pkl` | >790,000 protein–glycan binding interactions from >2,000 glycan-binding proteins. |
| `data/glycan_embedding/N_glycans_df.pkl` | N-glycan sequences for validation and benchmarking of embedding quality. |
| `data/glycan_embedding/df_species.pkl` | Species-level glycan association data. |

### Data Dimensions (approximate)

- Features (LC-MS signals): hundreds to thousands per experiment
- Samples: tens to low hundreds per cohort
- Glycan sequences in reference database: ~50,500
- Protein–glycan interactions: >790,000

---

## 5. Methodology

### Phase 1: Exploratory Data Analysis

**Goal:** Gain a thorough understanding of the dataset before any filtering or modelling. Identify data quality issues, instrument artefacts, and the biological structure of the data.

#### 1.1 Data Loading & Inspection

- Load `data_matrix.csv`, `acquisition_list.csv`, and `feature_metadata.csv` into pandas DataFrames
- Verify alignment between sample IDs in the data matrix and metadata
- Report: number of samples per class, number of features, proportion of missing values (zeros treated as missing in LC-MS), batch composition

#### 1.2 Isomer Detection

Isomers are molecules with the same molecular mass but different structures, which elute at different retention times. In LC-MS data, they appear as features with the same (or very similar) m/z but distinct retention times.

- Group features by rounded m/z (tolerance: ±0.01 Da)
- For each group, check whether multiple features exist with different RT values
- Report: count of features involved in isomeric pairs/groups
- Visualise: scatter plot of m/z vs RT, highlighting isomeric groups

#### 1.3 Isotope & Adduct Detection

Isotopes and adducts are features that arise from the same molecule but differ in mass by predictable increments (e.g., +1.003 Da for ¹³C isotope, +22 Da for Na adduct). They appear at the same retention time but different m/z values.

- Group features by retention time (tolerance: ±0.05 min)
- Within each RT group, calculate pairwise m/z differences
- Flag feature pairs matching common isotope/adduct offsets
- Report: proportion of features likely to be isotopes or adducts
- Visualise: RT-grouped m/z ladder plots for example molecules

#### 1.4 Correlation Analysis

- Compute pairwise Pearson correlations between all features across QC samples
- Isomers and isotopes of the same molecule should be highly correlated
- Generate a clustered correlation heatmap
- Identify modules of co-varying features (possible structural families)

#### 1.5 Feature Distribution Across m/z and Retention Time

- Histogram of m/z values across all features
- Histogram of retention times
- 2D density plot (m/z vs RT) — the "feature map"
- Annotate the expected glycan m/z range (>500 Da)
- Compare feature distributions across classes to detect systematic shifts

#### 1.6 Contamination Assessment

- Compare signal intensities in blank samples vs. biological samples feature-by-feature
- A feature with high blank-to-sample ratio is a likely contaminant
- Define contamination threshold: blank intensity > 30% of mean sample intensity → flag as contaminant
- Report: number and proportion of contaminated features

#### 1.7 Exogenous Standards Verification

- Load `exogenous_standards.csv`
- Match each standard to features in the data matrix by m/z and RT (within tolerance)
- Plot: detected intensity of each standard across all injections, coloured by batch
- Assess: are standards detected consistently? Do intensities drift across run order?
- This verifies instrument stability and the integrity of the analytical run

#### 1.8 Intensity Distribution & Run Order Effects

- Boxplots of total ion current (TIC = sum of all feature intensities) per sample, ordered by run order
- Colour by class and batch
- Look for systematic intensity drift across the run — a common LC-MS artefact
- PCA of all samples using raw data: colour by batch, then by class
- Assess whether batch effects dominate biological variation at this stage

#### 1.9 QC Coefficient of Variation (CV) and D-Ratio

For each feature:

- **CV on QC samples** = (std of QC intensities) / (mean of QC intensities) × 100%
  - Measures analytical reproducibility. Lower CV = more reproducible feature.
- **D-Ratio** = (std of QC intensities) / (std of all biological sample intensities)
  - Measures the ratio of technical to biological variation. D-Ratio < 0.4 is desirable (biological variation dominates).

- Plot: distribution of CVs across all features (histogram)
- Plot: CV vs D-Ratio scatter plot per feature
- Report: proportion of features with CV < 30% and D-Ratio < 0.4

---

### Phase 2: Data Quality Control & Filtering

**Goal:** Remove low-quality features and samples to produce a clean, analytically valid dataset for downstream analysis.

#### 2.1 Feature Filtering

Apply the following filters sequentially, tracking feature count at each step:

| Filter | Criterion | Rationale |
|--------|-----------|-----------|
| Analytical reproducibility | CV < 30% on QC samples | Removes features that are not reliably measured by the instrument |
| Prevalence | Detection rate ≥ 70% in at least one biological class | Removes features absent from most samples (uninformative or near noise) |
| Mass range | m/z > 500 Da | Focuses analysis on the glycan mass range; small molecules below this threshold are unlikely to be glycans |
| Contamination | Not flagged as contaminant in Step 1.6 | Removes background signals |

- Report: feature count before and after each filter step (waterfall bar chart)

#### 2.2 Sample Quality Check

- Calculate per-sample missingness (proportion of features below detection limit)
- Calculate per-sample TIC (total intensity)
- Flag outlier samples using PCA: Hotelling T² statistic > 95th percentile
- Visualise: PCA score plot with flagged samples annotated
- Decision: remove samples with > 50% missing features or extreme TIC outliers (document each removal)

#### 2.3 Missing Value Imputation

LC-MS data contains structural zeros (feature genuinely absent) and technical zeros (feature present but below detection limit). Strategy:

- For features with < 30% missing per class: impute with half the minimum observed value per feature (standard LC-MS practice for below-detection-limit values)
- For features with ≥ 30% missing per class: already removed by the prevalence filter in Step 2.1

#### 2.4 Data Transformation

- **Log2 transformation**: stabilises variance across the intensity range (heteroscedasticity correction)
- **Probabilistic Quotient Normalisation (PQN)** or **median normalisation**: corrects for sample-to-sample loading differences
- Visualise: boxplots of feature intensities per sample before and after transformation
- Confirm: reduced variance spread and improved inter-sample comparability

#### 2.5 Run-order Drift Assessment (if needed)

The analysis is restricted to batch 1, so inter-batch correction (e.g. ComBat) is not applicable. However, within a single batch, instrument signal can drift systematically across run order as the column degrades or detector sensitivity changes. This is assessed using the QC samples, which were injected at regular intervals specifically to track this effect.

- Plot mean QC intensity per feature across run order and fit a LOWESS trend
- If a significant trend is detected, apply QC-based signal correction: for each feature, divide sample intensities by the interpolated QC trend at the corresponding run order position
- Visualise: QC intensity vs run order before and after correction
- If no trend is detected, document this and proceed without correction

---

### Phase 3: Statistical & Discriminatory Analysis

**Goal:** Identify which features show statistically significant differences between disease groups, and characterise the multivariate structure of the data.

#### 3.1 Unsupervised Exploration

- **PCA** (Principal Component Analysis):
  - Apply to filtered, normalised data matrix (biological samples only)
  - Plot PC1 vs PC2, coloured by class and by batch
  - Report: variance explained per component (scree plot)
  - Assess: do classes separate along early PCs?

- **UMAP** (Uniform Manifold Approximation and Projection):
  - Non-linear embedding for visualising complex structure
  - Hyperparameters: n_neighbors=15, min_dist=0.1, n_components=2
  - Plot coloured by class, batch, and cohort

- **Hierarchical Clustering**:
  - Cluster samples using Ward linkage on Euclidean distance in feature space
  - Produce annotated heatmap (rows = samples, columns = top variable features)
  - Annotate with class labels to assess natural grouping

#### 3.2 Univariate Statistical Testing

For each feature, test whether expression differs across the three biological classes:

- **Normality check**: Shapiro-Wilk test per feature
- **Parametric**: one-way ANOVA (for features passing normality)
- **Non-parametric**: Kruskal-Wallis test (for features failing normality)
- **Multiple testing correction**: Benjamini-Hochberg False Discovery Rate (FDR) at q < 0.05
- **Pairwise comparisons** (post-hoc): Tukey HSD (parametric) or Dunn's test (non-parametric) for significant features
- **Effect size**: Cohen's f (ANOVA) or eta-squared (η²)

Report: number of significant features per comparison pair, their m/z and RT distribution

#### 3.3 Fold-Change Analysis & Volcano Plots

For each pairwise class comparison (Disease vs Control, Disease vs Benign, Benign vs Control):

- Calculate: log2 fold-change = log2(mean class A intensity) - log2(mean class B intensity)
- Plot: volcano plot — x-axis: log2 fold-change, y-axis: -log10(adjusted p-value)
- Annotate: features passing both fold-change (|log2FC| > 1) and significance (q < 0.05) thresholds
- Colour: up-regulated in disease (red), down-regulated (blue), not significant (grey)

#### 3.4 Multivariate Discriminant Analysis

- **PLS-DA** (Partial Least Squares Discriminant Analysis):
  - Supervised dimensionality reduction for class separation
  - Evaluate with permutation test (1000 permutations) to assess overfitting
  - Extract VIP (Variable Importance in Projection) scores per feature

- **Correlation heatmap of top features**:
  - Select top 50 features by combined VIP + ANOVA significance score
  - Cluster and visualise as annotated heatmap

---

### Phase 4: Machine Learning Classification

**Goal:** Build, compare, and interpret multi-class classifiers that predict disease state from glycan profiles.

#### 4.1 Experimental Setup

- **Data**: filtered, normalised, batch-corrected feature matrix (biological samples only)
- **Target**: class label (3 classes: Disease, Benign, Control)
- **Split**: stratified train/test split (75% / 25%), preserving class proportions and batch representation
- **Cross-validation**: stratified 5-fold CV on the training set
- **Metrics**: macro-averaged F1 score (primary), balanced accuracy, per-class AUC-ROC

#### 4.2 Models

| Model | Library | Notes |
|-------|---------|-------|
| Logistic Regression (L2) | scikit-learn | Baseline linear model; interpretable coefficients |
| Random Forest | scikit-learn | Ensemble of decision trees; built-in feature importance |

#### 4.3 Hyperparameter Tuning

- Use `GridSearchCV` (scikit-learn) for both models
- Tune on training set only (5-fold CV); no access to test set during tuning

Key hyperparameters to tune:
- Logistic Regression: regularisation strength C
- Random Forest: n_estimators, max_depth, min_samples_split

#### 4.4 Feature Importance

- Use Random Forest feature importances (mean decrease in impurity) to rank features
- Cross-reference top features with Phase 3 statistical results (VIP scores, volcano plot hits)
- This provides a consistent, interpretable ranking without the computational overhead of wrapper methods

#### 4.5 Model Evaluation

For each model, on the held-out test set:

- Per-class precision, recall, F1
- Macro-averaged AUC-ROC (one-vs-rest)
- ROC curve plot per class

Compare the two models in text; summary results reported in the key results table.

#### 4.6 Model Interpretability — SHAP Analysis

Apply SHAP (SHapley Additive exPlanations) to the best-performing model:

- **Summary plot (beeswarm)**: feature importance ranked by mean |SHAP value|, each dot = one sample
- **Bar plot**: mean absolute SHAP values per feature (global importance)
- Map top SHAP features back to m/z and RT values from `feature_metadata.csv`

---

### Phase 5: Glycan Embedding & Biological Interpretation

**Goal:** Use the `glycowork` library to enrich identified glycans with structural and biological context, generate a glycan embedding space, and interpret the molecular biology underlying the biomarker candidates.

#### 5.1 Glycan Sequence Loading & Annotation

- Load `data/glycan_embedding/glycan_list.csv`
- Each row is a glycan identified in the LC-MS experiment, annotated with:
  - IUPAC sequence notation (e.g., `Gal(b1-4)GlcNAc(b1-2)Man...`)
  - Monosaccharide composition: dHex (fucose), Hex (galactose/mannose), HexNAc (GlcNAc/GalNAc), Neu5Ac (sialic acid)
  - Tissue species and tissue sample type
- Summarise composition statistics: distributions of each monosaccharide count

#### 5.2 Biological Enrichment via glycowork

Using the `glycowork` library:

**Disease Association Enrichment**
- Cross-reference glycan sequences against `df_glycan.pkl` (reference database of ~50,500 glycans with disease labels)
- Identify which identified glycans have known associations with lung cancer, other cancers, or inflammatory diseases
- Use glycowork's built-in enrichment functions to test whether disease-associated glycans are overrepresented among the biomarker candidates

**Protein Binding Enrichment**
- Cross-reference against `glycan_binding.pkl` (>790,000 protein–glycan interactions)
- Identify which glycan-binding proteins interact with the top biomarker glycans
- Use glycowork to map interacting proteins to biological context (lectins, selectins, immune receptors)
- Assess biological relevance to lung cancer

**Species & Tissue Distribution**
- Cross-reference against `df_species.pkl`
- Identify whether top biomarker glycans are known to be expressed preferentially in lung tissue or tumour-associated tissues

#### 5.3 Glycan Embedding via glycowork

- Use glycowork's built-in embedding tools to generate feature vectors for the identified glycans
- Apply UMAP to the resulting feature matrix to produce a 2D visualisation
- Colour points by: monosaccharide composition, disease association, tissue type, sialic acid content
- For the top biomarker glycans, examine their nearest neighbours in the embedding space and assess whether they share disease associations or structural features

#### 5.4 Embedding Quality Assessment

- **N-glycan benchmark**: load `N_glycans_df.pkl` — known N-glycan sequences
- Check that N-glycans cluster together in the embedding space (they share structural features)
- Nearest-neighbour analysis: do the top biomarker glycans cluster with other disease-associated glycans?

---

### Phase 6: Visualization & Summary

**Goal:** Consolidate all results into a clear, reproducible set of figures and a summary narrative suitable for course submission.

#### 6.1 Figure Panel

| Figure | Content | Phase |
|--------|---------|-------|
| Fig 1 | Feature map (m/z vs RT), coloured by CV | Phase 1 |
| Fig 2 | CV distribution histogram; D-Ratio scatter | Phase 1 |
| Fig 3 | Standards intensity across run order | Phase 1 |
| Fig 4 | Feature count waterfall (filtering steps) | Phase 2 |
| Fig 5 | PCA before/after normalisation & batch correction | Phase 2 |
| Fig 6 | PCA + UMAP of cleaned data coloured by class | Phase 3 |
| Fig 7 | Volcano plots (3 pairwise comparisons) | Phase 3 |
| Fig 8 | Top features heatmap (samples × features) | Phase 3 |
| Fig 9 | SHAP beeswarm | Phase 4 |
| Fig 10 | Glycan embedding UMAP | Phase 5 |

Model comparison results, classification metrics, enrichment findings, and embedding neighbourhood analysis are reported in text and the results summary table rather than as standalone figures.

#### 6.2 Key Results Summary Table

| Category | Finding |
|----------|---------|
| Features after QC | X / Y original features retained (Z%) |
| Significant univariate features (q < 0.05) | — |
| Best classifier | — (macro F1: —, AUC: —) |
| Top 5 biomarker features | m/z values, RT, direction of change |
| Disease-associated glycans | Count and top associations |
| Protein interaction enrichment | Top proteins and pathways |

#### 6.3 Biological Interpretation Narrative

- Which glycan classes (sialylated, fucosylated, high-mannose, etc.) are most discriminatory?
- Are the top biomarker glycans previously implicated in lung cancer or other cancers?
- What protein interactions do they mediate — and what does this suggest about tumour biology?
- What are the limitations of the analysis (cohort size, batch effects, annotation coverage)?

---

## 6. Tools & Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.2.3 | Data loading, manipulation, and filtering |
| `numpy` | 2.2.1 | Numerical operations |
| `scipy` | 1.14.1 | Statistical tests (ANOVA, Kruskal-Wallis, Shapiro-Wilk) |
| `statsmodels` | 0.14.4 | Multiple testing correction (BH FDR), PLS-DA |
| `scikit-learn` | 1.6.0 | PCA, UMAP preprocessing, ML classifiers, metrics |
| `xgboost` | latest | Gradient boosting (available for supplementary comparison if time allows) |
| `shap` | latest | Model interpretability (SHAP values) |
| `umap-learn` | latest | Non-linear dimensionality reduction |
| `matplotlib` | 3.10.0 | Core plotting |
| `seaborn` | 0.13.2 | Statistical visualisation (heatmaps, violin plots) |
| `bokeh` | 3.9.0 | Interactive visualisations |
| `glycowork` | 1.7.1 | Glycan sequence analysis, enrichment, embedding |
| `glycorender` | 0.2.5 | Glycan structure rendering |
| `ipython` | 8.39.0 | Interactive notebook environment |

---

## 7. Expected Outcomes

By the end of this project, we expect to produce:

1. **A curated, analysis-ready glycomics dataset** — raw LC-MS data processed through rigorous QC, normalisation, and batch correction

2. **A ranked list of glycan biomarker candidates** — features statistically significant across multiple tests (ANOVA, volcano plot, PLS-DA VIP, SHAP), with m/z and RT annotations

3. **A comparative ML classification analysis** — Logistic Regression and Random Forest evaluated via 5-fold stratified CV, with results interpreted in terms of model behaviour, feature consistency, and separation between disease states rather than against a fixed benchmark

4. **SHAP-based molecular interpretations** — connecting model predictions to specific glycan features and their biological roles

5. **An annotated glycan embedding space** — a 2D UMAP representation of the identified glycans, coloured by composition, disease association, and protein interaction profile

6. **Enrichment insights** — identification of which disease-relevant proteins interact with the top biomarker glycans, suggesting candidate biological pathways

7. **A fully documented Jupyter notebook workflow** — reproducible, well-commented code for each analysis step

---

## 8. Project Timeline

| Week | Phase | Key Deliverables |
|------|-------|-----------------|
| Week 1 | Phase 1: EDA | Data loading complete; feature map, CV/D-Ratio, standards, contamination, and batch effect plots generated; isomer and isotope detection done |
| Week 2 | Phase 2: QC & Filtering | Filtered data matrix; missingness imputed; log2 + PQN normalisation applied; ComBat batch correction assessed; analysis-ready dataset saved |
| Week 3 | Phase 3: Statistical & Unsupervised Analysis | PCA and UMAP of cleaned data; hierarchical clustering; Kruskal-Wallis + FDR correction; volcano plots for all pairwise comparisons; candidate feature list finalised |
| Week 4 | Phase 4: ML Classification | Logistic Regression and Random Forest trained and evaluated; 5-fold CV results; feature importance ranking; SHAP beeswarm analysis |
| Week 5 | Phase 5: Glycan Embedding + Write-up | glycowork embedding and UMAP; disease and protein enrichment analysis; 4-page paper drafted; presentation slides prepared |

---

## 9. Documentation Plan

### Notebook Structure

Each analysis phase will be a separate Jupyter notebook section (or standalone `.ipynb` file), with the following documentation conventions:

**Cell-level documentation:**
- Markdown header cell at the start of each analysis block describing the goal and method
- Code cells: one short docstring per function explaining inputs, outputs, and purpose
- Output cells: a markdown cell below each major plot describing what is shown and what it means

**Variable naming:**
- DataFrames: descriptive snake_case names (`data_matrix`, `qc_samples`, `filtered_features`)
- Figures: saved to `figures/` subdirectory with descriptive filenames (`fig01_feature_map.png`)

**Reproducibility:**
- Set random seed at the top of each notebook (`np.random.seed(42)`)
- Save intermediate outputs (filtered matrix, normalised matrix) to `data/processed/`
- Record software versions in the first cell

**Final submission package:**
- `PROJECT_PLAN.md` — this document (project overview and methodology)
- `notebooks/` — one notebook per phase
- `figures/` — all saved figures, labelled by figure number
- `data/processed/` — intermediate processed files
- `requirements.txt` — package versions for reproducibility
- `README.md` — brief guide to running the notebooks in order

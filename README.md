# Corrode-Sense-ML: Predicting Residual Yield Force in Corroded Reinforcing Steel with Machine Learning

This repository provides a machine learning framework to predict the residual yield force (Fy) of corroded reinforcing steel bars, enabling non-destructive evaluation of structural integrity in reinforced concrete. Designed for civil engineers and researchers, it offers high-accuracy predictions, advanced visualizations, and support for custom datasets to assess corrosion-induced degradation in infrastructure.

## Project Overview

Corrosion reduces the yield force of reinforcing steel bars by up to 80%, driven by factors like penetration depth, mass loss, and exposure duration. Traditional empirical models (e.g., linear reductions per ACI 318) struggle with nonlinear effects such as pitting or fatigue. This project leverages six machine learning regression models, trained on 1,349 tensile tests from 26 global studies, to deliver precise Fy predictions for retrofits and risk assessments.

## Dataset Description

The `Data.csv` file, sourced from https://zenodo.org/records/8035720, compiles 1,349 tensile test records from studies spanning 1994 to 2023. It covers deformed bars (Grades 420-500 MPa, diameters 6-32 mm) under natural and simulated corrosion (e.g., electrolysis, immersion). Key features are outlined below.

| Feature | Description | Range/Stats | Importance |
|---------|-------------|-------------|------------|
| Mass Loss (%) | Weight lost to corrosion | 0-75% (mean: ~5%) | Primary degradation metric; correlates -0.7 with Fy |
| Penetration Depth (mm) | Depth of rust penetration | 0-1.3 mm (mean: 0.15 mm) | Drives nonlinear strength loss; >0.2 mm impacts ductility |
| Exposure Duration (hours) | Time under corrosive conditions | 0-14,610 (mean: 1,280) | Influences pitting in aggressive environments |
| Nominal Diameter (mm) | Bar diameter | 6-32 (mean: 15 mm) | Larger bars mitigate strength loss |
| Fy (kN) | Residual yield force | 2.8-417 (mean: 113 kN) | Target variable for structural safety |

Approximately 40% of samples represent natural corrosion (e.g., coastal exposures). Missing values (~5% in non-critical fields) are managed through imputation or exclusion. The dataset favors mild-to-moderate corrosion; extreme cases (>50% mass loss) are underrepresented, necessitating local validation.

## Machine Learning Models

Six regression models were chosen for their predictive performance, evaluated via 100-fold cross-validation (10x10 repeats) to ensure robustness. These models balance accuracy, efficiency, and applicability to corrosion’s nonlinear patterns.

| Category | Model | Strengths | Weaknesses | Use Case |
|----------|-------|-----------|------------|----------|
| Tree Ensembles | Gradient Boosting Regression Trees (GBRT) | Robust to outliers like pitting | Variance in small datasets | High-accuracy predictions (R² ~0.97) |
| | Random Forest (RF) | Stable via bagging | Less precise on sparse data | Quick baselines for noisy inputs |
| | CatBoost Regression (CBR) | Handles categorical features (e.g., bar origin) | Memory-intensive | Categorical and mixed data |
| Neural Networks | Artificial Neural Network (ANN) | Captures complex nonlinearities | Less interpretable | Interactions like load-corrosion |
| Kernel Machines | Gaussian Process Regression (GPR) | Probabilistic uncertainty estimates | Poor scaling for large datasets | Risk assessment with uncertainty |
| | Support Vector Regression (SVR) | Robust to noisy data | Sensitive to kernel parameters | Sparse or field-collected data |

### Performance Metrics
| Model | R² (Mean) | RMSE (kN) | MAE (kN) | MAPE (%) |
|-------|-----------|-----------|----------|----------|
| GBRT | 0.97 | 8.2 | 5.1 | 4.2 |
| CBR | 0.96 | 9.1 | 5.8 | 4.8 |
| RF | 0.95 | 10.3 | 6.4 | 5.5 |
| ANN | 0.94 | 11.2 | 7.0 | 6.1 |
| SVR | 0.93 | 12.0 | 7.5 | 6.5 |
| GPR | 0.92 | 12.8 | 8.1 | 7.0 |

GBRT leads with the highest R² and lowest RMSE, making it ideal for engineering applications. GPR provides uncertainty quantification, valuable for safety-critical scenarios.

## Getting Started

1. **Clone and Install** this repository and download the libraries mentioned in requirements.txt.

2. **Run the Pipeline**:
Execute `python run.py` to train and evaluate all models, producing a `model_perfs.csv` file (6 models × 100 folds) and visualization plots. Runtime is approximately 2 hours on a standard CPU.

3. **Custom Predictions**:
Append new data to `Data.csv` (e.g., 16 mm bar, 10% mass loss) and rerun, or modify `predict_single.py` for single-sample predictions.

4. **Exploratory Analysis**:
Use the included `explore_data.ipynb` Jupyter notebook for visualizations, such as Fy versus penetration depth distributions.

## Technical Details

- **Core Script**: `run.py` handles data loading, preprocessing (MinMax scaling), 80/20 train-test splits, model training, and evaluation.
- **Utilities**: `functions.py` supports feature scaling, performance metrics, and plotting.
- **Extensibility**: New models can be integrated by adapting the GBRT module structure.
- **Compute**: CPU-intensive; GPU acceleration is recommended for ANN and GPR.

## Results and Insights

Tree-based models (GBRT, CBR, RF) deliver superior accuracy, with GBRT achieving R²=0.97 and RMSE=8.2 kN. These models reduce prediction errors by 50% compared to empirical formulas, as validated in https://doi.org/10.1016/j.conbuildmat.2024.137023. Visualizations highlight residual scatterplots and feature importance (e.g., penetration depth as the dominant factor).

## Limitations and Recommendations

- **Data Bias**: The dataset is lab-heavy; field data for severe corrosion (>50% mass loss) is limited.
- **Validation**: Predictions should complement physical inspections.
- **Future Work**: Incorporate SHAP for interpretability or develop a web-based prediction tool.

## Dependencies
- pandas==1.4.4
- numpy==1.26.4
- tqdm==4.66.4
- tensorflow==2.16.1
- keras==3.3.3
- scikit-learn==1.0.2
- catboost==1.2.5
- matplotlib==3.7.2
- shap==0.42.1


## Related Resources

- Dataset: https://zenodo.org/records/8035720

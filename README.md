# Smartphone Usage Anomaly Detection

This project applies unsupervised anomaly detection to smartphone usage data and builds an interpretable workflow for identifying unusual user behavior patterns. The analysis was developed as part of a Big Data Mining course and is structured for academic submission.

## Problem

The goal is to detect abnormal user behavior from a multivariate smartphone usage dataset and translate those anomalies into interpretable behavioral categories. Instead of relying on a single variable, the project treats anomaly detection as a multivariate problem using sleep, stress, screen time, productivity, and caffeine intake together.

## Dataset

The final submission uses a reduced processed dataset that keeps only the variables used in the analysis:

- `sleep_hours`
- `stress_level`
- `screen_time`
- `productivity_score`
- `caffeine_intake`

The processed file also includes model outputs and final labels:

- `iso_anomaly`
- `lof_anomaly`
- `svm_anomaly`
- `anomaly_votes`
- `final_anomaly`
- `anomaly_type`

Files:

- `EDA.ipynb`: complete notebook with markdown explanations, code, visualizations, and precomputed outputs
- `smartphone_usage_processed.csv`: reduced dataset used for the final notebook
- `Smartphone_Usage_Productivity_Dataset_50000.csv`: original source dataset

## Methods

The notebook follows a full anomaly detection pipeline:

1. Data loading and feature selection
2. Missing value handling with median imputation
3. Feature scaling using `StandardScaler`
4. Unsupervised anomaly detection with:
   - Isolation Forest
   - Local Outlier Factor
   - One-Class SVM
5. Ensemble decision rule:
   - final anomaly if at least 2 out of 3 models agree
6. Taxonomy classification of detected anomalies into:
   - Health-Risk
   - Productivity Paradox
   - Behavioral Anomaly
7. PCA-based visualization and feature distribution analysis
8. Approximate feature importance using Random Forest on the ensemble label

## Results

Based on the current notebook output:

- Final anomalies detected: `2,180 / 50,000`
- Final anomaly rate: `4.36%`
- Dominant anomaly type: `Behavioral Anomaly`

The analysis shows that anomaly patterns are better explained by combinations of features than by any single variable alone. `screen_time`, `sleep_hours`, and `stress_level` are among the most influential variables in the final anomaly structure, while `productivity_score` shows the largest average separation between anomaly and normal groups.

## Reproducibility

To review the project:

1. Open `EDA.ipynb`
2. Run all cells, or use the saved outputs already included in the notebook

Python libraries used:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Project Value

This project demonstrates:

- unsupervised anomaly detection on behavioral data
- model comparison and ensemble design
- interpretable anomaly taxonomy construction
- feature-driven insight extraction for reporting
- end-to-end notebook presentation suitable for academic and portfolio use

## Key Skills

- Python for machine learning workflows
- Data preprocessing, feature selection, and input standardization
- Unsupervised anomaly detection using Isolation Forest, Local Outlier Factor, and One-Class SVM
- Ensemble design for improving prediction robustness across multiple models
- ML result evaluation through model comparison, overlap analysis, and taxonomy-based interpretation
- Feature-level interpretation using tree-based importance approximation
- Experiment implementation and reporting in Jupyter Notebook
- Building interpretable ML outputs for downstream analysis and decision support

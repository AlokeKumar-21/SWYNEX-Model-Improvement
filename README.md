# SWYNEX Internship — Task 3: Model Improvement

## Overview
This project improves and validates a Logistic Regression classifier using hyperparameter tuning and stratified cross-validation as part of the SWYNEX Machine Learning Internship.

## Dataset
- **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset
- **Samples:** 569
- **Features:** 30 numerical features
- **Classes:** Malignant and Benign

The dataset is loaded using scikit-learn's built-in `load_breast_cancer()` function.

## Methodology
1. Split the dataset into training (80%) and testing (20%) subsets using stratified sampling.
2. Established a Logistic Regression baseline using a `StandardScaler` pipeline.
3. Used `GridSearchCV` with stratified 5-fold cross-validation to tune regularization strength and class weighting.
4. Evaluated both models on the same held-out test set.
5. Compared Accuracy, Precision, Recall, F1-score and ROC-AUC.
6. Visualized the comparison and confusion matrices.

## Results

| Metric | Baseline | Tuned |
|---|---:|---:|
| Accuracy | 98.25% | 97.37% |
| Precision | 98.61% | 97.26% |
| Recall | 98.61% | 98.61% |
| F1-score | 98.61% | 97.93% |
| ROC-AUC | 99.54% | 99.57% |

## Findings
Hyperparameter tuning selected a regularization parameter of `C = 0.1`, with the `lbfgs` solver and no class weighting.

The tuned model achieved a slightly higher ROC-AUC, but its accuracy, precision and F1-score were lower than those of the baseline on the held-out test set. Recall remained unchanged.

Therefore, tuning did not improve overall test-set performance in this experiment. The baseline performed better on most metrics, demonstrating why model changes should be validated rather than assumed to be improvements.

## Project Files
- `SWYNEX_Model_Improvement.ipynb` — model training, tuning, evaluation and visualizations
- `baseline_results.csv` — results from the previous task
- `model_improvement_results.csv` — Task 3 baseline-versus-tuned comparison
- `requirements.txt` — Python dependencies

## Run Locally

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Open the notebook in VS Code or Jupyter and run the cells from top to bottom.

## Technologies
Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn and Jupyter Notebook.

## Internship
Completed as part of the SWYNEX Machine Learning Internship.

# Comparative Analysis of Machine Learning Models for Breast Cancer Classification with SHAP and LIME-Based Explainability

## 📌 Project Overview

This project presents a comparative analysis of six machine learning and deep learning models for breast cancer classification using the Breast Cancer Wisconsin Diagnostic dataset provided by Scikit-learn.

The primary objective is to evaluate the predictive performance of different models and investigate how individual input features influence their predictions using Explainable Artificial Intelligence (XAI) techniques.

The project integrates SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to improve the interpretability of model predictions.

Instead of evaluating models solely on accuracy, this study also examines precision, sensitivity, specificity, F1-score, ROC-AUC, and feature-level explanations.

## 🎯 Objectives

- Compare classical machine learning algorithms with deep learning architectures.
- Evaluate models using multiple classification metrics.
- Identify important features influencing model predictions.
- Generate global explanations using SHAP summary plots.
- Generate local explanations using LIME for individual predictions.
- Compare model performance on a common held-out test dataset.
- Use stratified cross-validation to assess the performance of classical machine learning models.
- Produce reproducible experimental outputs for further research.

## 🧠 Models Implemented

Six classification models are implemented in this project.

| No. | Model | Framework |
|---|---|---|
| 1 | Logistic Regression | Scikit-learn |
| 2 | Random Forest | Scikit-learn |
| 3 | XGBoost | XGBoost |
| 4 | Multilayer Perceptron (MLP) | PyTorch |
| 5 | One-Dimensional Convolutional Neural Network (1D-CNN) | PyTorch |
| 6 | Transformer Encoder Classifier | PyTorch |

### 1. Logistic Regression

Logistic Regression serves as a linear classification baseline. It estimates the probability of a sample belonging to the malignant class using a logistic function.

### 2. Random Forest

Random Forest combines multiple decision trees to capture nonlinear relationships between features and produce classification predictions.

### 3. XGBoost

XGBoost uses gradient-boosted decision trees to learn complex patterns in the dataset.

### 4. Multilayer Perceptron (MLP)

The MLP is a fully connected neural network consisting of hidden layers, ReLU activations, and dropout regularization.

### 5. One-Dimensional CNN

The 1D-CNN applies convolutional filters across the ordered feature vector to learn local patterns in the input measurements.

### 6. Transformer Encoder

The Transformer model uses self-attention to learn relationships between feature representations. Learnable feature-position embeddings distinguish the positions of the input features.

**Note:** CNNs and Transformers are experimental architectures for this tabular dataset. Their use does not guarantee better performance than conventional tabular machine learning models.

## 📊 Dataset

The project uses the Breast Cancer Wisconsin Diagnostic dataset available through Scikit-learn.

Dataset source:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

### Dataset Characteristics

| Property | Description |
|---|---|
| Total observations | 569 |
| Input features | 30 |
| Classification classes | 2 |
| Malignant samples | 212 |
| Benign samples | 357 |
| Feature types | Numerical measurements |
| Dataset access | Scikit-learn |

The dataset contains measurements computed from digitized images of fine needle aspirate samples of breast masses.

The features describe characteristics of cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

### Feature Categories

The 30 numerical features represent three measurement groups:

- Mean measurements
- Standard error measurements
- Worst or largest measurements

Each group contains ten characteristics of cell nuclei.

### Class Encoding

The original Scikit-learn dataset uses:

- `0` = Malignant
- `1` = Benign

For this project, the labels are remapped consistently:

- `0` = Benign
- `1` = Malignant

Malignant is treated as the positive class when calculating precision, recall, F1-score, and ROC-AUC.

This convention makes sensitivity to malignant cases explicit in the evaluation.

## ⚙️ Data Preprocessing

The following preprocessing steps are performed:

1. Load the dataset using Scikit-learn.
2. Convert the input features to numerical arrays.
3. Remap the target labels so malignant is the positive class.
4. Split the data into training and testing sets using stratified sampling.
5. Apply feature standardization for Logistic Regression and deep learning models.
6. Keep tree-based models on the original feature scale.

### Train-Test Split

The dataset is divided into:

- Training set: 80%
- Testing set: 20%

The split uses stratification to preserve the approximate class distribution.

A fixed random seed of `42` is used for reproducibility.

The scaler is fitted exclusively on the training data and then applied to the test data.

This prevents information from the held-out test set from influencing the scaling parameters.

## 📈 Evaluation Metrics

All six models are evaluated on the same held-out test set.

### 1. Accuracy

The proportion of correctly classified observations among all observations.

### 2. Precision

The proportion of observations predicted as malignant that are actually malignant.

### 3. Recall / Sensitivity

The proportion of actual malignant observations correctly identified by the model.

### 4. Specificity

The proportion of actual benign observations correctly identified by the model.

### 5. F1-Score

The harmonic mean of precision and recall for the malignant class.

### 6. ROC-AUC

The area under the Receiver Operating Characteristic curve, measuring how well the model separates malignant and benign observations across classification thresholds.

These metrics provide complementary perspectives on classification performance.

In particular, accuracy alone may not adequately represent performance when the costs of different classification errors are unequal.

## 🔍 Explainable Artificial Intelligence (XAI)

An important component of this project is understanding the factors influencing model predictions.

Two complementary explanation techniques are used: SHAP and LIME.

### SHAP

SHAP estimates feature contributions to a model's output using Shapley-value-based explanations.

The project generates SHAP summary plots to examine feature contributions across a selected subset of test observations.

The SHAP plots help investigate:

- Which features have the largest overall contributions.
- Whether higher or lower feature values tend to increase the explained model output.
- How feature contributions vary across observations.
- Whether different models rely on similar features.

SHAP is applied using:

- `shap.LinearExplainer` for Logistic Regression.
- `shap.TreeExplainer` for Random Forest and XGBoost.
- `shap.KernelExplainer` for the PyTorch neural networks.

For the neural networks, the prediction wrapper returns class probabilities so that Kernel SHAP can explain the malignant-class probability.

### LIME

LIME approximates model behavior locally around an individual observation.

The project generates one LIME explanation per model for a selected test observation.

The explanations identify features that locally support or oppose a particular class prediction.

Using the same selected test observation across all six models helps compare their local explanations.

### SHAP vs. LIME

| Property | SHAP | LIME |
|---|---|---|
| Full name | SHapley Additive exPlanations | Local Interpretable Model-agnostic Explanations |
| Primary purpose | Explain feature contributions to model outputs | Approximate model behavior around an individual observation |
| Explanation scope | Global summaries and local explanations | Primarily local explanations |
| Main output | SHAP values | Feature contribution weights |
| Main use in this project | Compare feature contributions across observations | Investigate individual predictions |

SHAP and LIME provide approximations or interpretations of model behavior. They do not establish causal relationships between features and cancer.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical computations |
| Pandas | Data manipulation and tabular results |
| Matplotlib | Visualization |
| Scikit-learn | Preprocessing, classical ML, metrics, and cross-validation |
| XGBoost | Gradient-boosted tree classification |
| PyTorch | Deep learning model implementation and training |
| SHAP | Feature contribution analysis |
| LIME | Local model explanations |

## 📁 Project Structure

After running the script, the project directory will resemble the following:

```text
XAI/
│
├── breast_cancer_xai_pytorch.py
├── README.md
│
└── outputs/
    ├── model_metrics.csv
    ├── model_metrics_table.png
    ├── classical_cv_metrics.csv
    ├── run_metadata.json
    │
    ├── confusion_matrix_LogisticRegression.png
    ├── confusion_matrix_RandomForest.png
    ├── confusion_matrix_XGBoost.png
    ├── confusion_matrix_MLP.png
    ├── confusion_matrix_1D-CNN.png
    ├── confusion_matrix_Transformer.png
    │
    ├── roc_curve_LogisticRegression.png
    ├── roc_curve_RandomForest.png
    ├── roc_curve_XGBoost.png
    ├── roc_curve_MLP.png
    ├── roc_curve_1D-CNN.png
    ├── roc_curve_Transformer.png
    │
    ├── shap_summary_LogisticRegression.png
    ├── shap_summary_RandomForest.png
    ├── shap_summary_XGBoost.png
    ├── shap_summary_MLP.png
    ├── shap_summary_1D-CNN.png
    ├── shap_summary_Transformer.png
    │
    ├── lime_explanation_LogisticRegression.png
    ├── lime_explanation_RandomForest.png
    ├── lime_explanation_XGBoost.png
    ├── lime_explanation_MLP.png
    ├── lime_explanation_1D-CNN.png
    ├── lime_explanation_Transformer.png
    │
    ├── mlp_state_dict.pt
    ├── 1d_cnn_state_dict.pt
    └── transformer_state_dict.pt

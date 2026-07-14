# HeartDisease_Prediction


Heart Disease Risk Prediction: Logistic Regression vs. ANN

A comparative machine learning case study evaluating a classical model (Logistic Regression) against a deep learning model (Artificial Neural Network / MLP) for predicting self-reported heart disease / heart attack risk from health survey data.


Academic use only. This project predicts patterns in self-reported survey data. It is not a clinical diagnostic tool and must not replace evaluation by a qualified healthcare professional.



Overview

This repository contains an end-to-end supervised learning workflow built as an NCIT Machine Learning group case study. It trains, tunes, and compares two model families on the same preprocessed dataset and test split, then packages the tuned models for reuse.

Pipeline covered:


Data acquisition and validation
Preprocessing without data leakage (median imputation + standard scaling fit only on training folds)
Exploratory data analysis
Baseline and hyperparameter-tuned Logistic Regression
Baseline and hyperparameter-tuned ANN (Keras Sequential MLP)
Structured hyperparameter search (GridSearchCV for Logistic Regression; a custom configuration sweep for the ANN)
Evaluation using Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, and PR-AUC
Confusion matrices, ROC curves, and precision-recall curves
Model persistence and a prediction interface for new records


Dataset


Source: Heart Disease Health Indicators Dataset (cleaned BRFSS 2015 data), via kagglehub
Original documentation: CDC BRFSS 2015
Target: HeartDiseaseorAttack — 0 = no reported coronary heart disease/heart attack, 1 = reported coronary heart disease/heart attack
Features (21): HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, Diabetes, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
The notebook can train on a reproducible stratified 5,000-row sample (USE_SAMPLE = True, default) or the full 253,680-record cleaned dataset (USE_SAMPLE = False).



Why Logistic Regression, not Linear Regression?

The target is binary (0/1), so Logistic Regression is used to estimate class probabilities. Linear Regression is designed for continuous-valued targets and is not appropriate here.

Models

Classical: Logistic RegressionDeep Learning: ANN (MLP)Libraryscikit-learnTensorFlow / KerasTuning methodGridSearchCV, 5-fold stratified CV, scored on ROC-AUCSweep over hidden-layer sizes, dropout, learning rate, L2 regularization, batch size, and class weightingRegularizationL1/L2 penalty, inverse-strength CDropout, L2 weight decay, early stoppingClass imbalance handlingOptional class_weight="balanced"Optional balanced class weights

Both models are evaluated as baseline (default settings) and tuned (best configuration found) versions on the same held-out test set, enabling a fair four-way comparison.

Evaluation Metrics

Each model variant is scored on:


Accuracy
Precision
Recall
F1-score
Specificity
ROC-AUC
PR-AUC (average precision)


Results are compiled into a master comparison table plus ROC and precision-recall curve plots overlaying all four model variants (Logistic Regression baseline/tuned, ANN baseline/tuned).

Getting Started

Run in Google Colab (recommended)


Upload Model_Test_Train.ipynb to Google Colab.
Select Runtime → Run all.
Keep USE_SAMPLE = True for a faster ~5,000-row experiment, or set it to False to train on the full ~253,680-record dataset.


Outputs produced by the training notebook

Running the full pipeline saves the following to an output directory (zipped for download):


tuned_logistic_regression.joblib — full classical pipeline (imputer + scaler + model)
tuned_ann.keras — trained ANN
ann_preprocessor.joblib — preprocessing pipeline for the ANN
model_comparison.csv — master metrics table
logistic_feature_coefficients.csv — feature coefficients for the tuned Logistic Regression model
ann_tuning_results.csv — results of the ANN configuration sweep
model_metadata.json — feature order, chosen threshold, best hyperparameters, and random seed, for reproducibility


Making predictions

prediction_model.ipynb loads the saved models (from Google Drive, by default) and demonstrates:


Loading the tuned Logistic Regression and ANN + its preprocessor
An interactive CLI-style prompt collecting all 21 required features
Running both models on a new record and returning predicted probabilities and classes


A new record must supply all 21 features in the exact order used during training; the prediction function validates this before scoring. The default classification threshold is 0.50, chosen for academic purposes only.

Requirements


Python 3
pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
tensorflow (Keras)
kagglehub (for dataset download)


All dependencies are installed automatically in the first cell of Model_Test_Train.ipynb when run in Colab.

Reproducibility

A fixed random seed (SEED = 42) is applied across Python's random, NumPy, and TensorFlow to keep data splits and training runs consistent across executions.

Limitations


The data is self-reported and observational (BRFSS survey), not clinically verified.
Strong test performance does not make either model a diagnostic tool.
The 0.50 decision threshold is an academic default, not a clinically validated operating point.


Authors
Bishowraj Chapai
Roshani Gautam
Aditya Raj Shah

# Predicting User Activity in Music Streaming Platforms through Early Behavioural Signals

This repository contains the code for the master’s thesis:
"Predicting User Activity in Music Streaming Platforms through Early Behavioural Signals: 
A Comparative Machine Learning Approach".

## Project Overview
This thesis investigates whether early behavioural signals can be used to predict future user 
activeness on a music-streaming platform. Using impression-level data from NetEase Cloud Music, 
user-level features are engineered from the first four actions after registration to capture 
interaction intensity, social richness, depth and persistence, and exposure to popular content.

Multiple machine learning models (Logistic Regression, Random Forest, MLP, XGBoost) are trained 
and evaluated, and model predictions are interpreted using SHAP.

## Repository Structure
•⁠  ⁠⁠ EDA/ ⁠ – Data extraction, preprocessing, exploratory data analysis, and feature engineering notebooks that transform raw data into user-level features  
•⁠  ⁠⁠ data/ ⁠ – Processed datasets (CSV files) including raw input data, cleaned impression-level data, and final user-level datasets ready for modeling  
•⁠  ⁠⁠ models/ ⁠ – Model training and evaluation  
•⁠  ⁠⁠ robustness_checks/ ⁠ – Sensitivity analyses over alternative activeness definitions

## Reproducibility
The main analyses reported in the thesis can be reproduced by running the scripts in the 
following order:
1.⁠ ⁠Data preprocessing and feature engineering (⁠ data/ ⁠)
2.⁠ ⁠Model training and evaluation (⁠ models/ ⁠)
3.⁠ ⁠Robustness and sensitivity analyses (⁠ robustness_checks/ ⁠)

Due to data access restrictions, the raw data are not publicly shared.

## Environment
Python 3.12.4  
Main libraries: pandas, numpy, scikit-learn, XGBoost, SHAP

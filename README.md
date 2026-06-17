# Loan Approval Prediction using Random Forest

## Overview

This repository contains my contribution to a team-based Loan Approval Prediction project. My responsibility was to develop, tune, and evaluate a Random Forest Classifier for predicting loan approval outcomes.

## My Contribution

* Trained a Random Forest Classifier on the preprocessed dataset.
* Performed hyperparameter tuning to improve model performance.
* Evaluated the model using classification metrics.
* Compared Random Forest results with a Decision Tree model.
* Analyzed model strengths, weaknesses, and overall performance.

## Model Configuration

The final Random Forest model was trained using:

* Number of Trees (n_estimators): 200
* Maximum Depth: 5
* Minimum Samples Split: 2
* Minimum Samples per Leaf: 1

## Results

The Random Forest model achieved an accuracy of **75%** on the test dataset.

### Classification Performance

| Class                  | Precision | Recall | F1-Score |
| ---------------------- | --------- | ------ | -------- |
| Not Approved (Class 0) | 62%       | 52%    | 56%      |
| Approved (Class 1)     | 80%       | 86%    | 83%      |

### Confusion Matrix

| Actual / Predicted | Not Approved | Approved |
| ------------------ | ------------ | -------- |
| Not Approved       | 16           | 15       |
| Approved           | 10           | 59       |

## Comparison with Decision Tree

* Decision Tree achieved 72% accuracy.
* Random Forest achieved 75% accuracy.
* Random Forest performed better overall and provided more balanced predictions.
* Random Forest was particularly effective at predicting approved loans.

## Key Findings

* Random Forest reduced overfitting compared to a single Decision Tree.
* Hyperparameter tuning improved prediction performance.
* Using multiple trees produced more reliable and stable results.
* The model handled loan approval prediction effectively despite class imbalance.

## Challenges Faced

* The dataset contained more approved loans than non-approved loans.
* Hyperparameter tuning required multiple experiments to achieve the best results.
* Categorical features had to be converted into numerical values before training.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook

## Project Report

A detailed report containing model results, observations, and conclusions is included in this repository.

## Note

This repository contains only my individual contribution to a team Loan Approval Prediction project. The complete project was developed collaboratively by all team members.

## **German Credit Data - Risk Analysis**


## Project Overview
This project aims to develop and compare machine learning models to predict whether a credit applicant is likely to default or not based on demographic, financial, and loan-related factors.
Using the German Credit dataset, the analysis explores logistic regression as a baseline model and applies regularized methods (Lasso, Ridge) and Random Forest to enhance prediction accuracy and interpretability.

The goal is to identify the most influential variables contributing to credit risk and evaluate model performance using AUC, accuracy, and confusion matrices.
- **Dataset Name:** German Credit Data

- **Authors:** Hoffman, P. and K. Bach

- **Publisher:** UCI Machine Learning Repository

- **URL:**<https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29>


## Key Results
- **Best AUC:** Random Forest (76%)
- **Fewest Errors:** XGBoost (48 total errors)
- **Feature Selection:** Lasso identified 25 critical features

## Models Compared
1. Logistic Regression (baseline)
2. Lasso Regression (L1 regularization)
3. Ridge Regression (L2 regularization)
4. Random Forest
5. XGBoost

## Key Steps

1. Data Cleaning & Preprocessing

- Handle missing values and duplicates.

- Convert categorical variables to dummy variables.

- Detect and handle outliers (where appropriate).

- Split the data into training and testing sets.

2. Exploratory Data Analysis (EDA)

- Visualize distributions and correlations.

- Understand variable importance and potential multicollinearity.

3. Modeling

- Logistic Regression (baseline).

- Lasso and Ridge (using glmnet and cv.glmnet).

- Random Forest and XGBoost for comparison.

4. Evaluation

- AUC-ROC

- Confusion matrix

- Visualize ROC curves and feature importance.


## Technologies Used
- R, tidyverse, caret
- glmnet (Lasso/Ridge)
- randomForest
- xgboost
- pROC, ggplot2


### Key Predictive Features (XGBoost)

The XGBoost model identified the following as most important for credit risk:

These findings align with domain knowledge in credit risk assessment.

| Rank | Feature              |    Gain   | Interpretation                                    |
| :--: | :------------------- | :-------: | :------------------------------------------------ |
|   1  | `checking_statusA14` | **0.235** | No checking account — *most important predictor!* |
|   2  | `duration`           |   0.133   | Loan duration (in months)                         |
|   3  | `credit_amount`      |   0.109   | Loan amount (in Deutsche Marks)                   |
|   4  | `age`                |   0.082   | Customer’s age                                    |
|   5  | `credit_historyA34`  |   0.053   | Credit history: “delay in paying off in the past” |
|   6  | `savingsA65`         |   0.048   | Savings: “unknown/no savings account”             |
|   7  | `purposeA41`         |   0.044   | Loan purpose: “used car”                          |
|   8  | `checking_statusA11` |   0.036   | Checking account: “< 0 DM” (negative balance)     |
|   9  | `installment_rate`   |   0.031   | Installment rate as % of disposable income        |
|  10  | `jobA174`            |   0.022   | Job type: “skilled employee/official”             |






Lasso regression identified **25 key predictive features** with only 2% AUC loss

**Reasons for selecting German Credit Risk data**

~1000 observations, 20 features → manageable.

Binary target: good vs. bad credit risk.

Contains both numerical (age, duration, amount) and categorical (housing, job, purpose) features → can practice handling real-world style data.

Well-documented and widely used, so if I get stuck, examples exist online.

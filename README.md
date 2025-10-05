**German Credit Data - Risk Analysis**


## Project Overview
Predicting credit default risk using machine learning models on the German Credit dataset.
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










## Technologies Used
- R, tidyverse, caret
- glmnet (Lasso/Ridge)
- randomForest
- xgboost
- pROC, ggplot2

**Reasons for selecting German Credit Risk data**

~1000 observations, 20 features → manageable.

Binary target: good vs. bad credit risk.

Contains both numerical (age, duration, amount) and categorical (housing, job, purpose) features → can practice handling real-world style data.

Well-documented and widely used, so if I get stuck, examples exist online.

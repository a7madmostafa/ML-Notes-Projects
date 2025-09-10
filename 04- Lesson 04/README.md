# Machine Learning: Evaluation & Tuning

This lesson covers the final stages of a machine learning project: evaluating model performance, validating it robustly using cross-validation, and tuning hyperparameters for the best results.

## Table of Contents
1.  [Regression Evaluation Metrics](#regression-evaluation-metrics)
2.  [Data Splitting Strategy](#data-splitting-strategy)
3.  [R² Score (Coefficient of Determination)](#r²-score-coefficient-of-determination)
4.  [Cross-Validation](#cross-validation)
5.  [Hyperparameter Tuning](#hyperparameter-tuning)

---

## Regression Evaluation Metrics

Metrics are used to quantify the performance of a regression model.

*   **Mean Squared Error (MSE):** The average of the squared differences between predicted and actual values. Heavily penalizes large errors.
    *   `MSE = (1/n) * Σ(y_actual - y_pred)²`
*   **Root Mean Squared Error (RMSE):** The square root of MSE. It is in the same units as the target variable, making it more interpretable.
    *   `RMSE = √(MSE)`
*   **Mean Absolute Error (MAE):** The average of the absolute differences. It gives a linear penalty, so it is less sensitive to outliers than MSE/RMSE.
    *   `MAE = (1/n) * Σ|y_actual - y_pred|`
*   **R² Score (Coefficient of Determination):** Measures the proportion of the variance in the target variable that is predictable from the features. (See detailed section below).

## Data Splitting Strategy

A proper data split is essential for a realistic evaluation of a model's performance.

*   **Training Data (~80%):** The data used to **train** the model (i.e., learn its parameters like weights).
*   **Validation Data:** A subset of the training data used for **model selection** and **hyperparameter tuning**. It helps check the model's performance during development without touching the test set.
*   **Testing Data (~20%):** The held-out data used for the **final evaluation** of the model. It provides an unbiased estimate of how the model will perform on new, unseen data.

## R² Score (Coefficient of Determination)

The R² score is a key metric for evaluating regression models. It answers the question: "How much better is my model than simply always predicting the mean of the target variable?"

*   **Formula:** `R² = 1 - (SS_res / SS_tot)`
    *   **Sum of Squares Residual (SS_res):** `Σ(y_actual - y_pred)²` - The total error of our model.
    *   **Sum of Squares Total (SS_tot):** `Σ(y_actual - y_mean)²` - The total error of a simple baseline model that always predicts the mean.
*   **Interpretation:**
    *   **R² = 1:** Perfect model. The predictions are exactly equal to the actual values. (Best scenario)
    *   **R² = 0:** The model is only as good as predicting the mean. (Worst case for a useful model)
    *   **R² < 0:** The model is **worse** than just predicting the mean. This indicates a severely flawed model.

## Cross-Validation

Cross-Validation (CV) is a robust technique for assessing how the results of a model will generalize to an independent dataset. It is crucial when the dataset is not very large.

*   **K-Fold Cross-Validation:** The standard method.
    1.  Shuffle the dataset and split it into `k` groups (folds).
    2.  For each unique fold:
        *   **Hold-out** one fold as the **validation set**.
        *   **Train** the model on the remaining `k-1` folds.
        *   **Evaluate** the model on the held-out fold and save the score.
    3.  The final performance is the **average of all k scores**. Common values for `k` are 5 or 10.
*   **Leave-One-Out Cross-Validation (LOOCV):** A special case where `k` is set to the number of observations in the dataset. This is very computationally expensive but can be useful for **very small datasets**.

## Hyperparameter Tuning

Hyperparameters are configuration settings for the model that are not learned from the data but are set before the training process. Finding the best ones is called tuning.

*   **Hyperparameters vs. Parameters:**
    *   **Parameters:** Internal to the model, learned from the data (e.g., weights in Linear Regression).
    *   **Hyperparameters:** External, set by the user before training (e.g., polynomial degree, regularization strength `alpha`, learning rate).

### Tuning Strategies:

1.  **Grid Search**
    *   **How it works:** Define a grid of all possible hyperparameter values you want to try. The algorithm trains and evaluates a model for **every single combination** using cross-validation.
    *   **Pros:** Exhaustive; you are guaranteed to find the best combination within the grid.
    *   **Cons:** Computationally very expensive, especially with a large grid.

2.  **Random Search**
    *   **How it works:** Define a search space for each hyperparameter. The algorithm selects **random combinations** from this space for a fixed number of iterations and evaluates them using cross-validation.
    *   **Pros:** Often finds a good combination much faster than Grid Search.
    *   **Cons:** Not exhaustive; might miss the absolute best combination, but often finds a very good one efficiently.
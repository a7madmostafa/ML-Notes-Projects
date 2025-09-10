# Machine Learning: Linear Models & Regularization

This document summarizes advanced concepts in Linear Regression, including multiple and polynomial regression, the bias-variance tradeoff, and regularization techniques.

## Table of Contents
1.  [Linear Regression Types](#linear-regression-types)
2.  [Normal Equation vs. Gradient Descent](#normal-equation-vs-gradient-descent)
3.  [Implementation in Scikit-Learn](#implementation-in-scikit-learn)
4.  [Gradient Descent Variants](#gradient-descent-variants)
5.  [Multiple Linear Regression](#multiple-linear-regression)
6.  [Polynomial Regression](#polynomial-regression)
7.  [Bias-Variance Tradeoff](#bias-variance-tradeoff)
8.  [Regularization](#regularization)
9.  [L1 vs. L2 Penalty](#l1-vs-l2-penalty)

---

## Linear Regression Types

*   **Simple Linear Regression:** Models the relationship between one independent variable and a target.
*   **Multiple Linear Regression:** Models the relationship between multiple independent variables and a target.
*   **Polynomial Regression:** Models non-linear relationships by adding polynomial terms of the independent variables.

## Normal Equation vs. Gradient Descent

| Factor | Normal Equation | Gradient Descent |
| :--- | :--- | :--- |
| **Solution Type** | Analytical, Closed-Form | Iterative Numerical Approximation |
| **Iterations** | No iterations needed | Requires many iterations (> 10^5 possible) |
| **Learning Rate** | Not needed | Requires careful tuning of learning rate |
| **Complexity** | O(n³) - Slow for many features (n) | O(kn²) - Efficient for large n |
| **Feature Scaling** | Not required | Required |
| **Global Solution** | Guaranteed | May converge to local minimum (but convex problems guarantee global) |

## Implementation in Scikit-Learn

### LinearRegression (Normal Equation)
*   **Class:** `sklearn.linear_model.LinearRegression`
*   **Method:** Uses the Ordinary Least Squares (OLS) method to minimize the residual sum of squares.
*   **Solution:** Computes weights directly using the matrix formula: `W = (X.T * X)^-1 * X.T * y`

### SGDRegressor (Gradient Descent)
*   **Class:** `sklearn.linear_model.SGDRegressor`
*   **Method:** Linear model fitted by minimizing a regularized empirical loss with Stochastic Gradient Descent (SGD).
*   **Process:** The gradient of the loss is estimated per sample, and the model is updated with a decreasing learning rate.

## Gradient Descent Variants

The type of Gradient Descent is defined by how much data is used to compute the gradient in each iteration:

*   **Batch Gradient Descent:** Uses **all data points** to calculate the gradient for one update. Can be slow for very large datasets.
*   **Stochastic Gradient Descent (SGD):** Uses **one randomly chosen data point** to calculate the gradient for each update. Much faster but noisier.
*   **Mini-Batch Gradient Descent:** A compromise that uses **a random subset (mini-batch)** of data points for each update. This is the most common approach in practice.

## Multiple Linear Regression

*   **Model:** Extends simple linear regression to multiple features (`x₁, x₂, ..., xₙ`).
*   **Equation:**
    `y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ`
*   **Vectorized Form:** This can be written succinctly using a weight vector `W` and feature vector `X` (with `x₀ = 1` for the bias term `w₀`):
    `y = WᵀX`
    where `W = [w₀, w₁, ..., wₙ]` and `X = [1, x₁, x₂, ..., xₙ]`

## Polynomial Regression

*   **Purpose:** To fit non-linear data by creating new features that are polynomial combinations of the original features.
*   **Implementation:** Use `sklearn.preprocessing.PolynomialFeatures`
    *   **Example:** For a 2D input `[a, b]` and `degree=2`, it creates the new feature vector `[1, a, b, a², ab, b²]`.
*   **Model:** A linear regression model is then fit on these new polynomial features.
    *   The model equation becomes non-linear in the original inputs (e.g., `y = w₀ + w₁x₁ + w₂x₂ + w₃x₁² + w₄x₁x₂ + w₅x₂²`).

## Bias-Variance Tradeoff

A fundamental concept in ML that describes the tension between a model's simplicity and its accuracy.

*   **Bias Error:** Error due to overly simplistic assumptions in the model. A high-bias model **underfits** the training data.
    *   *Symptoms:* High error on both training and test sets.
*   **Variance Error:** Error due to excessive complexity, making the model overly sensitive to noise in the training data. A high-variance model **overfits** the training data.
    *   *Symptoms:* Low error on training set, high error on test set.
*   **Goal:** Find the optimal model complexity that minimizes total error by balancing bias and variance.

## Regularization

Regularization techniques prevent overfitting by adding a penalty term to the model's cost function, discouraging overly complex models with large weights.

The general form of the regularized cost function is:
`J(w) = MSE + Penalty Term`

### Types of Regularization:

1.  **Ridge Regression (L2 Regularization)**
    *   **Penalty Term:** `α * Σ(w_i²)` (Sum of squared weights)
    *   **Effect:** Shrinks weights towards zero but never exactly to zero. Good for handling correlated features.
    *   **Scikit-Learn:** `sklearn.linear_model.Ridge`

2.  **Lasso Regression (L1 Regularization)**
    *   **Penalty Term:** `α * Σ|w_i|` (Sum of absolute weights)
    *   **Effect:** Can shrink less important feature weights exactly to zero. Performs **automatic feature selection**.
    *   **Scikit-Learn:** `sklearn.linear_model.Lasso`

3.  **Elastic Net**
    *   **Penalty Term:** A mix of L1 and L2 penalties. Combines the properties of both Ridge and Lasso.
    *   **Formula:** `α * ρ * Σ|w_i| + [α * (1-ρ)/2] * Σ(w_i²)`
    *   **Scikit-Learn:** `sklearn.linear_model.ElasticNet`

## L1 vs. L2 Penalty

| Aspect | L2 Regularization (Ridge) | L1 Regularization (Lasso) |
| :--- | :--- | :--- |
| **Penalty Term** | `α * Σ |w|₂²` | `α * Σ |w|₁` |
| **Effect on Weights** | Shrinks weights proportionally. **Weights never become zero.** | Can force weights to be **exactly zero**. |
| **Feature Selection** | No | Yes (automatic) |
| **Use Case** | When all features are likely to be important. | When you suspect many features are irrelevant and want feature selection. |
| **Robustness** | Handles correlated features well. | Can struggle with highly correlated features (may select one arbitrarily). |
# Machine Learning: Data Preprocessing 
 
This document covers essential data preprocessing techniques, including handling categorical variables, missing values, and feature scaling. These are critical steps before training a machine learning model. 
 
## Table of Contents 
1.  [Handling Categorical Data](#handling-categorical-data) 
2.  [Handling Missing Values](#handling-missing-values) 
3.  [Feature Scaling](#feature-scaling) 
4.  [The Preprocessing Workflow](#the-preprocessing-workflow) 
 
--- 
 
## Handling Categorical Data 
 
Most machine learning algorithms require numerical input. Categorical data (text labels) must be converted into numbers. 
 
### Common Encoding Techniques: 
 
1.  **Ordinal Encoding**  
    *   **Use Case:** For categorical variables with a natural, ordered relationship (e.g., "low", "medium", "high").  
    *   **Method:** Assigns an integer to each category (e.g., low=0, medium=1, high=2) based on the order.  
    *   **Scikit-Learn Class:** `sklearn.preprocessing.OrdinalEncoder`  
 
2.  **One-Hot Encoding (OHE)**  
    *   **Use Case:** For nominal categorical variables (no intrinsic order), like "red", "blue", "green".  
    *   **Method:** Creates new binary (0/1) columns for each category. A value of 1 indicates the presence of that category.  
        *   Example: For a "Color" feature with values "Red" and "Blue", it creates two new columns: `Color_Red` and `Color_Blue`.  
        *   A red item would be encoded as `Color_Red=1`, `Color_Blue=0`.  
    *   **Limitation:** If the feature has **many unique categories** (e.g., thousands of zip codes), OHE produces a very large sparse matrix, leading to high memory usage and slower training.  
    *   **Scikit-Learn Class:** `sklearn.preprocessing.OneHotEncoder`  
 
3.  **Binary Encoding**  
    *   **Use Case:** For **nominal variables with high cardinality** (many unique categories) where One-Hot Encoding would create too many columns.  
    *   **Method:**  
        1. Each category is assigned an integer label.  
        2. The integer is converted into its binary representation.  
        3. Each digit of the binary number becomes a new column.  
        * Example: Suppose we have 5 categories: A=1, B=2, C=3, D=4, E=5. Their binary forms are 001, 010, 011, 100, 101 → producing 3 binary columns instead of 5 OHE columns.  
    *   **Advantages:**  
        * Reduces dimensionality compared to OHE.  
        * Handles high-cardinality features efficiently.  
    *   **Library Support:** `category_encoders.BinaryEncoder` (not in core scikit-learn, but widely used).  
 
---
 
## Handling Missing Values 
 
Real-world datasets often have missing values. It is crucial to handle them before training a model, as most algorithms cannot process them directly. 
 
Common strategies include:  
*   **Deletion:** Removing rows or columns with missing values. This is simple but can lead to loss of valuable data.  
*   **Imputation:** Filling in missing values with a statistic calculated from the available data.  
    *   **Numerical Features:** Use the **mean** or **median** of the feature.  
    *   **Categorical Features:** Use the **mode** (most frequent value) of the feature.  
*   **Scikit-Learn Class:** `sklearn.impute.SimpleImputer`  
 
---
 
## Feature Scaling 
 
Many machine learning algorithms perform better or converge faster when features are on a similar scale. This is especially important for:  
*   Algorithms that rely on distance calculations (e.g., K-Nearest Neighbors, K-Means clustering).  
*   Models that use gradient descent for optimization (e.g., Linear Regression, Neural Networks).  
 
### Common Scaling Techniques: 
 
1.  **Normalization (Min-Max Scaling)**  
    *   Scales features to a fixed range, usually [0, 1].  
    *   **Formula:**  
      ```
      X_scaled = (X - X_min) / (X_max - X_min)
      ```  
    *   **Scikit-Learn Class:** `sklearn.preprocessing.MinMaxScaler`  
 
2.  **Standardization (Z-score Normalization)**  
    *   Scales features to have a mean of 0 and a standard deviation of 1.  
    *   **Formula:**  
      ```
      X_scaled = (X - μ) / σ
      ```  
    *   This is less affected by outliers than Min-Max scaling.  
    *   **Scikit-Learn Class:** `sklearn.preprocessing.StandardScaler`  
 
---
 
## The Preprocessing Workflow 
 
A standard workflow for applying these transformations correctly and avoiding data leakage is crucial. 
 
1.  **Split the Data:** Always split your data into **training** and **testing** sets first (`train_test_split`). The test set must remain completely unseen during the model training process.  
2.  **Fit on Training Data:** Calculate the necessary statistics for imputation and scaling **only from the training set**.  
3.  **Transform Both Sets:** Use the parameters (e.g., mean, min, max) learned from the training set to transform both the training and the test set.  
 
**Golden Rule:** Never fit your preprocessor on the test set. This prevents information from the test set from leaking into the training process, which would optimistically bias your model's performance evaluation.  

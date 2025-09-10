# Introduction to Machine Learning

This document summarizes key concepts from an introductory course on Machine Learning (ML).

## Table of Contents
1.  [What is Artificial Intelligence?](#what-is-artificial-intelligence)
2.  [A Brief History of AI](#a-brief-history-of-ai)
3.  [Why AI is Relevant Now](#why-ai-is-relevant-now)
4.  [What is Machine Learning?](#what-is-machine-learning)
5.  [ML vs. Traditional Programming](#ml-vs-traditional-programming)
6.  [AI, ML, DL, and Data Science](#ai-ml-dl-and-data-science)
7.  [Types of Machine Learning](#types-of-machine-learning)
8.  [Linear Regression](#linear-regression)
9.  [Gradient Descent](#gradient-descent)

---

## What is Artificial Intelligence?

AI (Artificial Intelligence) is the simulation of natural intelligence in machines. It is concerned with building systems that can:
*   **Reason**
*   **Discover Meaning**
*   **Learn from Experience**
*   **Generalize** knowledge to new situations

## A Brief History of AI

*   **1950:** Alan Turing proposes the "Turing Test".
*   **1956:** The field is officially born at the **Dartmouth Conference**.
*   **1966:** The first chatbot, **ELIZA**, is created.
*   **1972:** The first intelligent robot, **WABOT-1**, is built.
*   **1974-1980:** The **First AI Winter**.
*   **1980s:** Rise of **Expert Systems**.
*   **1987-1993:** The **Second AI Winter**.
*   **1997:** **IBM's Deep Blue** becomes the first computer to beat a world chess champion.
*   **2002:** iRobot's **Roomba** vacuum cleaner brings AI into homes.
*   **2011:** **IBM's Watson** wins the quiz show Jeopardy!.
*   **2014:** Chatbot **Eugene Goostman** is claimed to have passed the Turing test.
*   **2015:** **Amazon Echo** is released.

## Why AI is Relevant Now

The current AI boom is driven by four key factors:
1.  **Processing Power**
2.  **Data Growth**
3.  **Algorithm Development**
4.  **Storage Cost**

## What is Machine Learning?

ML is a subset of AI focused on the idea that systems can **learn from data** and **improve with experience**.

**Key Definitions:**
*   Arthur Samuel (1959): "Field of study that gives computers the ability to learn without being explicitly programmed."
*   Tom Mitchell (1997): "A computer program is said to learn from experience `E` with respect to some class of tasks `T` and performance measure `P`, if its performance at tasks in `T`, as measured by `P`, improves with experience `E`."

## ML vs. Traditional Programming

| Traditional Programming | Machine Learning |
| :--- | :--- |
| Data + Rules → Answer | Data + Answers → Model |
| The programmer defines the logic. | The algorithm finds the patterns. |

## AI, ML, DL, and Data Science

*   **Artificial Intelligence (AI):** The broad goal of creating intelligent machines.
*   **Machine Learning (ML):** The primary approach to achieving AI.
*   **Deep Learning (DL):** A powerful subfield of ML using neural networks.
*   **Data Science (DS):** A field that uses scientific methods to extract insights from data.
*   **Big Data (BD):** Characterized by the **5 Vs**: Volume, Velocity, Variety, Veracity, Value.

## Types of Machine Learning

1.  **Supervised Learning:** Learning from **labeled data**.
    *   **Regression:** Predicting a continuous value.
    *   **Classification:** Predicting a discrete category.

2.  **Unsupervised Learning:** Finding patterns in **unlabeled data**.
    *   **Clustering:** Grouping similar data points.
    *   **Dimensionality Reduction:** Compressing data.
    *   **Anomaly Detection:** Identifying unusual data points.

3.  **Reinforcement Learning:** Learning through **trial and error**.
    *   An **agent** takes **actions** in an **environment** to maximize a **reward**.

# Introduction to Machine Learning

## Linear Regression

### Simple Linear Regression
*   Models the relationship between **one feature (x)** and a **continuous target variable (y)**.
*   The model is represented by a linear equation:
    \[ y = w_0 + w_1 x \]
    *   `y`: Target / Output variable
    *   `x`: Feature / Input variable
    *   `w_0`: Y-intercept (Bias term)
    *   `w_1`: Slope (Weight for feature `x`)
*   The goal is to find the **Best Fit Line** that minimizes the error between the predicted and actual values.
*   This is achieved by finding the **Best Weights** (`w_0`, `w_1`).

### Key Concepts for Linear Regression
*   **Linear Target:** The variable you are trying to predict (e.g., house price).
*   **Best Fit Line:** The line that results in the smallest total prediction error.
*   **Objective:** Minimize the total error (cost) across all data points.

### Cost Function: Mean Squared Error (MSE)
*   The cost function `J` measures the performance of the model. For Linear Regression, Mean Squared Error is commonly used.
*   It is the average of the squared differences between the actual (`y_actual`) and predicted (`y_pred`) values.
    \[ J = \frac{1}{m} \sum_{i=1}^m \left( y_{actual}^{(i)} - y_{pred}^{(i)} \right)^2 \Rightarrow MSE \]
*   The objective of the learning algorithm is to **minimize** this cost function `J`.
*   A lower MSE means a better fit for the model (e.g., `J = 20` is better than `J = 50`).

## Gradient Descent

### Overview
*   Gradient Descent is an iterative optimization algorithm used to find the **optimal weights** that **minimize the cost function**.
*   It is a fundamental algorithm for training many machine learning models, including Linear Regression.

### The Gradient Descent Process
The algorithm follows these steps:

1.  **Random Initialization:** Start with random values for the weights (e.g., `w_0 = 0`, `w_1 = 0`).
2.  **Compute Gradient:** Calculate the partial derivative of the cost function `J` with respect to each weight. This gradient points in the direction of the steepest ascent.
    \[ \frac{\partial J}{\partial w_0}, \frac{\partial J}{\partial w_1} \]
3.  **Update Weights:** Adjust the weights by moving them a small step in the *opposite* direction of the gradient (towards the minimum).
    *   The update rule for each weight is:
        \[ w^{new} = w^{old} - \eta \frac{\partial J}{\partial w} \]
    *   `η` (eta) is the **Learning Rate**, a crucial hyperparameter that controls the size of the step.
4.  **Iterate:** Repeat steps 2 and 3 until the gradients approach zero (`∂J/∂w ≈ 0`), indicating that the algorithm has (likely) found the minimum of the cost function.

### The Role of the Learning Rate (η)
*   The learning rate controls how big the steps are during each update.
*   A good learning rate is essential for convergence:
    *   **Too Small:** The algorithm will be slow to converge.
    *   **Too Large:** The algorithm may overshoot the minimum and fail to converge or even diverge.

### Vectorized Form
*   Weights and features can be represented as vectors for a more general form:
    *   Weight vector: `[W] = [w_0, w_1]`
    *   Feature vector (with added bias term): `[X] = [1, x]`
*   The prediction can be written as the dot product: `y_pred = W^T · X`
*   The cost function and gradient descent update can be expressed using these vectors.

### Normal Equation (Closed-Form Solution)
*   The **Normal Equation** (or **Ordinary Least Squares (OLS)** solution) is an analytical method to directly calculate the optimal weights without the need for iteration.
*   It involves solving the equation where the partial derivatives of the cost function are set to zero:
    \[ \begin{aligned}
    &\frac{\partial J}{\partial w_0} = 0 \\
    &\frac{\partial J}{\partial w_1} = 0
    \end{aligned} \]
*   This provides a **closed solution** for the weights `w_0` and `w_1`.
*   **Comparison:** While Gradient Descent is an iterative algorithm, the Normal Equation solves for the weights directly. It is efficient for small datasets but can be computationally expensive for large ones.
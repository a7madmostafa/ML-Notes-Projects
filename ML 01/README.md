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

## Linear Regression

A fundamental algorithm used for **regression** tasks. It models the relationship between a target (`y`) and one or more features (`X`).

**Simple Linear Regression:**
*   Equation: $y = w_0 + w_1x$
*   **Goal:** Find the **best-fit line** that minimizes error.

## Gradient Descent

An optimization algorithm used to find the **optimal weights** that minimize the **Cost Function**.

**Cost Function (Mean Squared Error - MSE):**
$$J(w) = \frac{1}{m} \sum_{i=1}^m (y_{actual}^{(i)} - y_{pred}^{(i)})^2$$

**How Gradient Descent Works:**
1.  Initialize weights randomly.
2.  Calculate the gradient (derivative) of the cost function.
3.  Update weights: $w^{new} = w^{old} - \eta \frac{\partial J}{\partial w}$
4.  Repeat until convergence.

**Normal Equation:** An analytical method to directly calculate the optimal weights.

# 📊 Financial Data Analysis and Modeling

This Python script performs three main machine learning tasks using financial data:

1. **Clustering** companies based on key financial ratios using KMeans.
2. **Classifying** companies into profitable vs non-profitable using a Random Forest classifier.
3. **Predicting revenues** using a Linear Regression model.

The data is loaded directly from a CSV file hosted on GitHub, and all visualizations are saved locally in a `figures` directory.

---

## 1. 🔗 Clustering with PCA and KMeans

### Goal:

Group companies into **4 clusters** using financial indicators:

* Profit Margin
* Debt Ratio
* Return on Assets (ROA)

### Steps:

* Normalize the data using **StandardScaler**.
* Reduce dimensionality with **Principal Component Analysis (PCA)**.
* Cluster the companies with **KMeans**.

### 📐 Mathematical Background:

#### Principal Component Analysis (PCA):

PCA transforms the dataset into a lower-dimensional space:

$$
Z = XW
$$

Where:

* $X \in \mathbb{R}^{n \times p}$ is the normalized input matrix,
* $W$ contains the top eigenvectors of the covariance matrix $\text{Cov}(X)$,
* $Z$ is the projected data in reduced dimensions.

#### KMeans Clustering:

KMeans tries to minimize the within-cluster sum of squares:

$$
\min_{\{C_k\}} \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2
$$

Where:

* $C_k$ is the set of points in cluster $k$,
* $\mu_k$ is the centroid of cluster $k$.

---

## 2. ✅ Classification: Random Forest

### Goal:

Predict whether a company is **profitable or not** (binary classification) based on:

* Profit Margin
* Debt Ratio
* ROA

### Steps:

* Split data into training and test sets.
* Train a **RandomForestClassifier**.
* Evaluate with accuracy, confusion matrix, and classification report.

### 📐 Mathematical Background:

A **Random Forest** is an ensemble of decision trees:

* Each tree is trained on a random subset of the data.
* Final prediction is made by **majority vote** among the trees.

Given features $X$ and labels $y$, the model approximates:

$$
\hat{y} = \text{majority_vote} \left( T_1(X), T_2(X), \dots, T_n(X) \right)
$$

Each $T_i$ is a decision tree trained on a bootstrap sample.

---

## 3. 📈 Regression: Revenue Prediction

### Goal:

Predict a company’s **Revenues** based on:

* Assets
* Liabilities
* Net Income or Loss

### Steps:

* Split data into train/test.
* Train a **Linear Regression** model.
* Evaluate using **Mean Squared Error (MSE)**.

### 📐 Mathematical Background:

**Linear Regression** assumes a linear relationship:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3
$$

Where:

* $\hat{y}$ is the predicted revenue,
* $x_1 = \text{Assets}$, $x_2 = \text{Liabilities}$, $x_3 = \text{NetIncomeLoss}$,
* $\beta$'s are learned via least squares minimization:

$$
\min_{\beta} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

---

## 📂 Outputs:

All plots are saved to a local directory called `figures`:

* Clusters: `clusters.png`
* Confusion matrix: `confusion_matrix.png`
* Revenue prediction: `revenue_prediction.png`

---

## ✅ Summary

This pipeline shows how to:

* **Cluster** companies based on financial features.
* **Classify** profitability.
* **Predict** revenue values.

It combines unsupervised, supervised, and regression-based learning and is built on **scikit-learn**, using standard tools from the machine learning ecosystem.


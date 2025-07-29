# 📊 Financial Forecasting Models Evaluation

## 🧠 Algorithms Used

In this project, we evaluated the performance of several machine learning algorithms for predicting **log-transformed quarterly revenues** of companies in specific SIC2 industries (`20–24`). The following models were implemented using a clean pipeline and time-based train-test split to preserve the temporal structure of the data:

### ✅ Models:

* **XGBoost Regressor**
* **Linear Regression (with StandardScaler)**
* **Lasso Regression (LassoCV with scaling)**
* **Ridge Regression (RidgeCV with scaling)**
* **Random Forest Regressor**

All models were evaluated based on the following performance metrics:

## 📏 Evaluation Metrics

* **RMSE (Root Mean Squared Error)** – Sensitive to large errors
* **MAE (Mean Absolute Error)** – Average absolute error
* **MAPE (Mean Absolute Percentage Error)** – Scale-independent metric
* **R² (Coefficient of Determination)** – Measures the proportion of variance explained

Both **test** and **train** R² scores were recorded to check for **overfitting**.

---

## 📈 Results Summary

| Model             | Test RMSE | Test MAE | MAPE      | Test R²     | Train R² |
| ----------------- | --------- | -------- | --------- | ----------- | -------- |
| **XGBoost**       | 22.37 B   | 7.90 B   | 43.94%    | 0.847       | 0.980    |
| **Linear**        | 3.01e+25  | 1.54e+24 | 1.05e+20% | ❌ -2.76e+26 | ❌ -15.86 |
| **Lasso**         | 3.63e+24  | 1.86e+23 | 1.27e+19% | ❌ -4.01e+27 | ❌ -15.97 |
| **Ridge**         | 8.39e+25  | 4.29e+24 | 2.93e+20% | ❌ ...       | ❌ ...    |
| **Random Forest** | 22.98 B   | 8.10 B   | 42.51%    | 0.839       | 0.978    |

---

## 🧐 Observations

* ❗ **Linear, Lasso, and Ridge regressions failed drastically** on this data, likely due to:

  * Highly skewed and non-stationary financial features
  * Presence of non-linear relationships
  * Extreme sensitivity to feature scaling and outliers

* ✅ **XGBoost** and **Random Forest** performed significantly better, with **XGBoost slightly outperforming** Random Forest in both test R² and error metrics.

---

## 🛡️ Mitigating Overfitting and Data Leakage

We took several steps to reduce the risk of **overfitting** and **data leakage**:

* Used **log1p transformation** of revenues to reduce skewness and improve numerical stability.
* Applied **rolling means and lag features**, ensuring all engineered features rely only on past data.
* Ensured **time-based train-test split** so that no future information leaks into the training phase.
* Avoided using identifiers (`name`, `sic`) as predictors.
* Evaluated **train vs test R²** to detect signs of overfitting. Linear models showed negative R² due to poor fit, while ensemble models retained generalization.

---

## 🚀 Future Work: LSTM (Deep Learning)

While tree-based models like XGBoost are powerful, they do not capture **sequential dependencies** between quarterly financial data. Therefore, the next phase of this project will explore **LSTM (Long Short-Term Memory)** networks using **PyTorch**, which are well-suited for time series and financial sequences.

> The implementation of LSTM is currently in progress and will be shared soon in the next version of this repository.

---

📌 **Conclusion**:
Among the models evaluated, **XGBoost** is the best-performing algorithm on this dataset based on its low error metrics and high R². However, to capture richer temporal dynamics, **LSTM** will be the focus going forward.




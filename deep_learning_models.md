# NetIncomeLoss Prediction Project

We started this project by using a **Random Forest** model to predict `NetIncomeLoss` and achieved excellent results:
https://github.com/AZFARHAD24511/ML/blob/main/financial_analyses_ML_vf.ipynb


- **Key feature**: `lag_5` (NetIncomeLoss value from 5 periods ago)  
- **Outcome**:  
  - **R² ≈ 0.92**  
  - **MAE and RMSE** in the billions  
  - Fast training and evaluation times  


Given this success, we hypothesized that a time‑series model like **LSTM** could harness longer dependencies and deliver even better performance. We then:

1. **Combined multiple lags** (`lag_1` through `lag_5`)  
2. **Added rolling statistics** (`rolling_mean_4`, `rolling_std_4`)  
3. **Scaled the target** with `log1p` and `PowerTransformer`  
4. **Increased network depth** (two LSTM layers with 64 and 32 units)  
5. **Applied Dropout and EarlyStopping** to prevent overfitting  

Despite fully implementing these enhancements and running extensive experiments:

- **Training/Validation Loss** on the log scale showed only minor improvements  
- **R²** on the original scale: **≈ 0.08**  
- **RMSE**: roughly 9.1 billion  
- The LSTM struggled to capture the specific short‑term seasonal patterns, resulting in **underfitting**.

---


 ```python
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. Load & basic preprocess
df = pd.read_csv(
    "https://raw.githubusercontent.com/AZFARHAD24511/datasets/refs/heads/main/financial_dataset.csv"
)
df['ReportDate'] = pd.to_datetime(
    df['Year'].astype(str) + '-' + (df['Quarter']*3 - 2).astype(str) + '-01'
)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# 2. Feature engineering: categorical codes & financial ratios
df['SIC2']      = df['sic'].astype(str).str[:2].astype(int)
df['stprinc']   = df['stprinc'].astype('category').cat.codes
df['DebtRatio'] = df['Liabilities'] / (df['Assets'] + 1e-6)
df['ROA']       = df['NetIncomeLoss'] / (df['Assets'] + 1e-6)

# 3. Create lag_1 … lag_5
for k in range(1, 6):
    df[f'lag_{k}'] = df['NetIncomeLoss'].shift(k)

# 4. Rolling statistics (window=4)
df['rolling_mean_4'] = df['NetIncomeLoss'].rolling(window=4).mean()
df['rolling_std_4']  = df['NetIncomeLoss'].rolling(window=4).std()

# 5. Drop initial NaNs from lags/rolling and unused columns
df = (
    df
    .drop(columns=['Label','name','Year','Quarter'])
    .dropna()
    .sort_values('ReportDate')
    .reset_index(drop=True)
)

# 6. Define features and raw target
features = [c for c in df.columns if c not in ['NetIncomeLoss','ReportDate']]
X = df[features].values               # shape: (n_samples, n_features)
y_raw = df['NetIncomeLoss'].values     # raw target
dates = df['ReportDate']

# 7. Scale X and transform y with PowerTransformer
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

pt = PowerTransformer(method='yeo-johnson')
y_trans = pt.fit_transform(y_raw.reshape(-1,1)).flatten()

# 8. Sequence builder that returns both transformed and raw y
def make_sequences(X_arr, y_t, y_r, dates, window):
    Xs, ys_t, ys_r, ds = [], [], [], []
    for i in range(window, len(X_arr)):
        Xs.append(X_arr[i-window:i])
        ys_t.append(y_t[i])
        ys_r.append(y_r[i])
        ds.append(dates.iloc[i])
    return np.array(Xs), np.array(ys_t), np.array(ys_r), np.array(ds)

window = 8
X_seq, y_seq_trans, y_seq_raw, date_seq = make_sequences(
    pd.DataFrame(X_scaled), pd.Series(y_trans), pd.Series(y_raw), dates, window
)

# 9. Train/test split by date
cutoff = pd.to_datetime('2023-01-01')
mask_train = date_seq < cutoff
X_train, y_train_t, y_train_r = X_seq[mask_train], y_seq_trans[mask_train], y_seq_raw[mask_train]
X_test,  y_test_t,  y_test_r  = X_seq[~mask_train], y_seq_trans[~mask_train], y_seq_raw[~mask_train]
dates_test = date_seq[~mask_train]

# 10. Build deep LSTM with two layers
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window, X_train.shape[2])),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 11. Train with EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train_t,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# 12. Predict & inverse-transform y
y_pred_t = model.predict(X_test).flatten()
y_pred = pt.inverse_transform(y_pred_t.reshape(-1,1)).flatten()
y_true = y_test_r

# 13. Evaluation
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
print(f"\n✅ Final LSTM with multiple lags & rolling features:")
print(f"R²   = {r2:.4f}")
print(f"RMSE = {rmse:,.2f}")

# 14. Plot training vs validation loss
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train MSE')
plt.plot(history.history['val_loss'], label='Val MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# 15. Plot actual vs predicted
plt.figure(figsize=(8,4))
plt.plot(dates_test, y_true, label='Actual')
plt.plot(dates_test, y_pred, label='Predicted')
plt.xlabel('Report Date')
plt.ylabel('NetIncomeLoss')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
```


Epoch 1/50
1132/1132 ━━━━━━━━━━━━━━━━━━━━ 16s 10ms/step - loss: 1.4201 - val_loss: 0.9036
Epoch 2/50
1132/1132 ━━━━━━━━━━━━━━━━━━━━ 19s 9ms/step - loss: 0.4628 - val_loss: 0.8805
Epoch 3/50
1132/1132 ━━━━━━━━━━━━━━━━━━━━ 21s 10ms/step - loss: 1.3588 - val_loss: 0.8692
Epoch 4/50
1132/1132 ━━━━━━━━━━━━━━━━━━━━ 20s 10ms/step - loss: 0.3285 - val_loss: 0.8644
Epoch 5/50
1132/1132 ━━━━━━━━━━━━━━━━━━━━ 21s 11ms/step - loss: 0.3138 - val_loss: 0.9116
Epoch 6/50
1132/1132 ━━━━━━━━━━━━━━━━━━━━ 20s 10ms/step - loss: 0.9936 - val_loss: 0.8628
Epoch 7/50
1132/1132 ━━━━━━━━━━━━━━━━━━━━ 20s 10ms/step - loss: 0.5212 - val_loss: 0.8772
Epoch 8/50
1132/1132 ━━━━━━━━━━━━━━━━━━━━ 20s 10ms/step - loss: 0.5065 - val_loss: 0.8616
Epoch 9/50
1132/1132 ━━━━━━━━━━━━━━━━━━━━ 20s 10ms/step - loss: 2.1227 - val_loss: 0.8869
Epoch 10/50
1132/1132 ━━━━━━━━━━━━━━━━━━━━ 11s 10ms/step - loss: 0.5327 - val_loss: 0.8847
Epoch 11/50
1132/1132 ━━━━━━━━━━━━━━━━━━━━ 21s 10ms/step - loss: 0.9901 - val_loss: 0.9027
Epoch 12/50
1132/1132 ━━━━━━━━━━━━━━━━━━━━ 19s 9ms/step - loss: 0.4770 - val_loss: 0.8970
Epoch 13/50
1132/1132 ━━━━━━━━━━━━━━━━━━━━ 22s 10ms/step - loss: 0.7484 - val_loss: 0.9095
224/224 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step

✅ Final LSTM with multiple lags & rolling features:

R²   = 0.084

RMSE = 9,116,752,142.39





## Conclusion

- **Random Forest**, using simple lag features, explained over **90%** of the variance in `NetIncomeLoss`.  
- **LSTM**, despite its theoretical power, underperformed on this type of short‑term, seasonal financial data.  
- For datasets with large scale, high variance, and short-term seasonal dependencies, **tree‑based models** like Random Forest or XGBoost remain a robust and efficient baseline.

> **Final Recommendation**  
> Stick with tree‑based methods for this use case. Reserve LSTM and other deep time‑series architectures for problems with long‑term dependencies and continuous data streams.  


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

st.title("📊 SHAP Analysis - Streamlit Version of Your Original Code")

uploaded_file = st.file_uploader("Upload your semicolon-separated CSV file", type=["csv"])
if uploaded_file:
    df_fixed = pd.read_csv(
        uploaded_file,
        sep=r';(?=(?:[^"]*"[^"]*")*[^"]*$)',
        engine='python',
        quotechar='"',
        encoding='utf-8-sig'
    )

    df_fixed.columns = [col.replace('"', '').strip() for col in df_fixed.columns]
    df_fixed = df_fixed.applymap(lambda x: x.replace('"', '').strip() if isinstance(x, str) else x)

    st.success(f"Loaded {df_fixed.shape[0]} rows, {df_fixed.shape[1]} columns.")
    st.write(df_fixed.head())

    sample_n = st.slider("Select sample size", min_value=100, max_value=len(df_fixed), value=500, step=100)

    if st.button("Run SHAP Analysis"):
        with st.spinner("Running analysis..."):

            X = df_fixed.drop(columns=['y'])
            y = df_fixed['y'].map({'no': 0, 'yes': 1})

            X_enc = X.copy()
            for col in X_enc.select_dtypes(include=['object']).columns:
                X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))

            mi = mutual_info_classif(X_enc, y, discrete_features='auto', random_state=42)
            mi_series = pd.Series(mi, index=X_enc.columns).sort_values(ascending=False)

            st.subheader("Top 10 Features by Mutual Information")
            st.write(mi_series.head(10))

            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf.fit(X_enc, y)

            explainer = shap.Explainer(rf, X_enc)
            X_small = X_enc.sample(n=sample_n, random_state=42)
            shap_result = explainer(X_small)

            st.subheader("SHAP Summary Plot (Bar)")
            shap.summary_plot(shap_result.values[..., 1], X_small, plot_type='bar')
            st.pyplot(plt.gcf())

            st.subheader("SHAP Summary Plot (Beeswarm)")
            shap.summary_plot(shap_result.values[..., 1], X_small)
            st.pyplot(plt.gcf())

            st.subheader("SHAP Dependence Plot for 'duration'")
            if 'duration' in X_small.columns:
                shap.dependence_plot('duration', shap_result.values[..., 1], X_small)
                st.pyplot(plt.gcf())
            else:
                st.warning("'duration' column not found.")

st.markdown(\"\"\"\n---\n<p style='text-align:center; font-size:small;'>\nThis software was developed by Dr. Farhadi, PhD in Econometrics and Data Science. For any personal or commercial use, please cite the author.\n</p>\n\"\"\", unsafe_allow_html=True)

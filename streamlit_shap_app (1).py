import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="SHAP Analysis App", layout="wide")
st.title("📊 SHAP Analysis Explorer")

st.markdown(
    "Upload your CSV file, select a sample size, and explore feature importances via SHAP."
)

# 1. File upload
uploaded_file = st.file_uploader("Upload your semicolon-separated CSV file", type=["csv"])

if uploaded_file is not None:
    # read CSV preserving original code logic
    df_fixed = pd.read_csv(
        uploaded_file,
        sep=r';(?=(?:[^"]*"[^"]*")*[^"]*$)',
        engine='python',
        quotechar='"',
        encoding='utf-8-sig'
    )

    st.success(f"Loaded data with {df_fixed.shape[0]} rows and {df_fixed.shape[1]} columns.")
    st.write(df_fixed.head())

    # 2. Sample size selection (user-defined)
    max_n = df_fixed.shape[0]
    sample_n = st.slider(
        "Select sample size for SHAP computation", min_value=100, max_value=max_n,
        value=min(500, max_n), step=100
    )

    if st.button("Run SHAP Analysis"):
        with st.spinner("Computing feature selection and SHAP..."):
            # original code begins here
            df_fixed.columns = [col.replace('"', '').strip() for col in df_fixed.columns]
            df_fixed = df_fixed.applymap(lambda x: x.replace('"', '').strip() if isinstance(x, str) else x)

            # Separate features and target
            X = df_fixed.drop(columns=['y'])
            y = df_fixed['y'].map({'no': 0, 'yes': 1})

            # Encode categoricals
            X_enc = X.copy()
            for col in X_enc.select_dtypes(include=['object']).columns:
                X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))

            # 1️⃣ Univariate: Mutual Information
            mi = mutual_info_classif(X_enc, y, discrete_features='auto', random_state=42)
            mi_series = pd.Series(mi, index=X_enc.columns).sort_values(ascending=False)
            st.subheader("Top 10 Features by Mutual Information")
            st.write(mi_series.head(10))

            # 2️⃣ Model-based: RandomForest + SHAP
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf.fit(X_enc, y)

            explainer = shap.Explainer(rf, X_enc)
            X_small = X_enc.sample(n=sample_n, random_state=42)
            shap_result = explainer(X_small)

            # build class-1 Explanation
            values_c1 = shap_result.values[..., 1]
            base_c1 = shap_result.base_values[1]
            data = shap_result.data
            fnames = shap_result.feature_names
            from shap import Explanation
            shap_c1 = Explanation(values=values_c1,
                                  base_values=base_c1,
                                  data=data,
                                  feature_names=fnames)

            # SHAP bar chart
            st.subheader("Feature Importance (SHAP Bar)")
            fig1 = plt.figure()
            shap.plots.bar(shap_c1, show=False)
            st.pyplot(fig1)

            # SHAP beeswarm
            st.subheader("Beeswarm Plot")
            fig2 = plt.figure()
            shap.plots.beeswarm(shap_c1, show=False)
            st.pyplot(fig2)

            # Dependence for 'duration'
            st.subheader("Dependence Plot for 'duration'")
            if 'duration' in X_enc.columns:
                fig3 = plt.figure()
                shap.plots.scatter(shap_c1[:, 'duration'], show=False)
                st.pyplot(fig3)
            else:
                st.info("Column 'duration' not found for dependence plot.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:small;'>"
    "This software was developed by Dr. Farhadi, PhD in Econometrics and Data Science. "
    "For personal or commercial use, please cite the author."
    "</p>", unsafe_allow_html=True
)

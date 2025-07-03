import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

st.title("📊 SHAP Analysis - Streamlit Version (Fixed)")

uploaded_file = st.file_uploader("Upload your semicolon-separated CSV file", type=["csv"])

if uploaded_file:
    # Read file as text
    content = uploaded_file.getvalue().decode('utf-8')
    lines = content.strip().split('\n')

    # Parse header and rows manually
    header = lines[0].strip().split(';')
    rows = [line.strip().split(';') for line in lines[1:]]

    df_fixed = pd.DataFrame(rows, columns=header)

    # Clean double quotes from columns and data
    df_fixed.columns = [col.replace('"', '').strip() for col in df_fixed.columns]
    df_fixed = df_fixed.applymap(lambda x: x.replace('"', '').strip() if isinstance(x, str) else x)

    st.success(f"Loaded {df_fixed.shape[0]} rows, {df_fixed.shape[1]} columns.")
    st.write(df_fixed.head())

    sample_n = st.slider("Select sample size", min_value=100, max_value=len(df_fixed), value=500, step=100)

    if st.button("Run SHAP Analysis"):
        with st.spinner("Running analysis..."):

            if 'y' not in df_fixed.columns:
                st.error("❌ Column 'y' not found. Available columns:")
                st.write(df_fixed.columns.tolist())
                st.stop()

            X = df_fixed.drop(columns=['y'])
            y = df_fixed['y'].map({'no': 0, 'yes': 1})

            # Encode categorical features
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

st.markdown("""
---
<p style='text-align:center; font-size:small;'>
This software was developed by Dr. Farhadi, PhD in Econometrics and Data Science.
For any personal or commercial use, please cite the author.
</p>
""", unsafe_allow_html=True)

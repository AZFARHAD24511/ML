import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="SHAP Explorer", layout="wide")

st.title("📊 SHAP Feature Importance Explorer")
st.markdown(
    "Upload a semicolon-separated CSV file, select a sample size, and explore feature importances via SHAP."
)

# 1. File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=';', engine='python', quotechar='"')
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    st.success(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    st.write(df.head())

    # 2. Sample size selection
    max_n = df.shape[0]
    sample_n = st.slider(
        "Select sample size for SHAP computation", min_value=100, max_value=max_n,
        value=min(500, max_n), step=100
    )

    # 3. Run analysis button
    if st.button("Run SHAP Analysis"):
        with st.spinner("Computing SHAP values..."):
            # Clean quotes
            df.columns = [col.replace('"', '').strip() for col in df.columns]
            df = df.applymap(lambda x: x.replace('"', '').strip() if isinstance(x, str) else x)

            if 'y' not in df.columns:
                st.error("Target column 'y' not found in the data.")
                st.stop()

            # Prepare features and target
            X = df.drop(columns=['y'])
            y = df['y'].map({'no': 0, 'yes': 1})

            # Encode categorical features
            X_enc = X.copy()
            for col in X_enc.select_dtypes(include=['object']).columns:
                X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))

            # Fit RandomForest
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf.fit(X_enc, y)

            # SHAP Explainer
            explainer = shap.Explainer(rf, X_enc)
            X_small = X_enc.sample(n=sample_n, random_state=42)
            shap_result = explainer(X_small)

            # Extract class-1 explanations
            values_c1 = shap_result.values[..., 1]
            base_c1 = shap_result.base_values[1]
            data = shap_result.data
            fnames = shap_result.feature_names
            from shap import Explanation
            shap_c1 = Explanation(values=values_c1,
                                  base_values=base_c1,
                                  data=data,
                                  feature_names=fnames)

            # Plot SHAP bar
            st.subheader("Feature Importance (SHAP Bar)")
            fig1 = plt.figure()
            shap.plots.bar(shap_c1, show=False)
            st.pyplot(fig1)

            # Plot SHAP beeswarm
            st.subheader("Beeswarm Plot")
            fig2 = plt.figure()
            shap.plots.beeswarm(shap_c1, show=False)
            st.pyplot(fig2)

            # Dependence plot for 'duration'
            st.subheader("Dependence Plot for 'duration'")
            if 'duration' in X_enc.columns:
                fig3 = plt.figure()
                shap.plots.scatter(shap_c1[:, 'duration'], show=False)
                st.pyplot(fig3)
            else:
                st.info("Column 'duration' not found.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: small;'>"
    "This software was developed by Dr. Farhadi, PhD in Econometrics and Data Science. "
    "For any personal or commercial use, please cite the author."
    "</p>", unsafe_allow_html=True
)

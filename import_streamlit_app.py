import streamlit as st
import runpy

st.title("SHAP Analysis App")

uploaded = st.file_uploader("Upload your Python analysis script", type="py")
if uploaded:
    # ذخیرهٔ موقت فایل:
    with open("user_script.py", "wb") as f:
        f.write(uploaded.getbuffer())
    st.success("Script uploaded. Now run it below.")
    if st.button("Run Analysis"):
        # اجرای اسکریپت خودت
        runpy.run_path("user_script.py", run_name="__main__")
else:
    st.info("Please upload your analysis .py file.")

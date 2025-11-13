import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🏗️", layout="wide")

# ---- LOAD MODEL ----
try:
    with open("concrete_model.pkl", "wb") as f:
        model, scaler = pickle.load(f)
    model_loaded = True
except Exception as e:
    st.error("⚠️ Could not load model. Make sure 'concrete_model.pkl' is in the same directory.")
    model_loaded = False

# ---- HEADER ----
st.title("🏗️ Concrete Compressive Strength Prediction App")
st.markdown("This app predicts **Concrete Compressive Strength (csMPa)** based on the mixture composition.")

st.divider()

# ---- SIDEBAR ----
st.sidebar.header("👤 Developer Info")
st.sidebar.markdown("""
**Name:** Tanvi Bramhnakar  
📧 **Email:** [tanvibramhnakar18@gmail.com](mailto:tanvibramhnakar18@gmail.com)  
💻 **GitHub:** [github.com/tanvibramhnakar7](https://github.com/tanvibramhnakar7)  
🔗 **LinkedIn:** [linkedin.com/in/tanvi-bramhnakar-4b1285294](https://www.linkedin.com/in/tanvi-bramhnakar-4b1285294)
""")

# ---- INPUT SECTION ----
st.header("🧱 Input Concrete Mixture Details")

col1, col2, col3 = st.columns(3)

with col1:
    cement = st.number_input("Cement (kg/m³)", min_value=0.0, step=1.0)
    slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, step=1.0)
    flyash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, step=1.0)

with col2:
    water = st.number_input("Water (kg/m³)", min_value=0.0, step=1.0)
    superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, step=0.1)
    coarseaggregate = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0, step=1.0)

with col3:
    fineaggregate = st.number_input("Fine Aggregate (kg/m³)", min_value=0.0, step=1.0)
    age = st.number_input("Age (days)", min_value=1, step=1)

st.divider()

# ---- PREDICTION ----
if st.button("🔮 Predict Concrete Strength"):
    input_data = np.array([[cement, slag, flyash, water, superplasticizer,
                            coarseaggregate, fineaggregate, age]])
    
    if model_loaded:
        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            st.success(f"✅ Predicted Concrete Compressive Strength: **{prediction[0]:.2f} MPa**")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
    else:
        st.warning("Model not loaded. Please check your 'concrete_model.pkl' file.")

# ---- FOOTER ----
st.markdown("---")
st.caption("📘 *Developed by Tanvi Bramhnakar | Data Science & Machine Learning Enthusiast*")

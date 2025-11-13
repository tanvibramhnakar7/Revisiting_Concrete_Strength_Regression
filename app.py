import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------
# APP CONFIGURATION
# ---------------------------
st.set_page_config(page_title="Concrete Strength Prediction", page_icon="🧱", layout="wide")

st.title("🧱 Concrete Compressive Strength Prediction App")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Concrete_Data_Yeh.csv")
    df.columns = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 
                  'coarse_aggregate', 'fine_aggregate', 'age', 'strength']
    return df

df = load_data()

# ---------------------------
# SIDEBAR INFO
# ---------------------------
st.sidebar.header("👤 Python Developer")
st.sidebar.markdown("""
**Name:** Tanvi Bramhnakar
**LinkedIn:** [LinkedIn Profile](https://www.linkedin.com)  
**GitHub:** [GitHub Profile](https://github.com)  
**Gmail:** tanvibramhnakar18@gmail.com
""")

# ---------------------------
# DATA PREVIEW
# ---------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ---------------------------
# VISUALIZATION SECTION
# ---------------------------
st.subheader("📈 Data Insights")
st.write("Correlation Heatmap of Features:")
st.dataframe(df.corr())

# ---------------------------
# MODEL TRAINING
# ---------------------------
X = df.drop('strength', axis=1)
y = df['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
gb_model = GradientBoostingRegressor()

lr_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# ---------------------------
# SIDEBAR INPUTS
# ---------------------------
st.sidebar.header("🔧 Enter Material Values")
cement = st.sidebar.number_input("Cement (kg/m³)", 0.0, 540.0, 200.0)
slag = st.sidebar.number_input("Blast Furnace Slag (kg/m³)", 0.0, 360.0, 100.0)
flyash = st.sidebar.number_input("Fly Ash (kg/m³)", 0.0, 200.0, 50.0)
water = st.sidebar.number_input("Water (kg/m³)", 100.0, 250.0, 150.0)
superplasticizer = st.sidebar.number_input("Superplasticizer (kg/m³)", 0.0, 30.0, 5.0)
coarseagg = st.sidebar.number_input("Coarse Aggregate (kg/m³)", 800.0, 1200.0, 900.0)
fineagg = st.sidebar.number_input("Fine Aggregate (kg/m³)", 500.0, 1000.0, 700.0)
age = st.sidebar.number_input("Age (days)", 1, 365, 28)

# ---------------------------
# PREDICTION
# ---------------------------
input_data = np.array([[cement, slag, flyash, water, superplasticizer, coarseagg, fineagg, age]])

if st.sidebar.button("🔮 Predict Concrete Strength"):
    lr_pred = lr_model.predict(input_data)[0]
    gb_pred = gb_model.predict(input_data)[0]

    st.success(f"**Linear Regression Prediction:** {lr_pred:.2f} MPa")
    st.success(f"**Gradient Boosting Prediction:** {gb_pred:.2f} MPa")

    # Model performance metrics
    lr_r2 = r2_score(y_test, lr_model.predict(X_test))
    gb_r2 = r2_score(y_test, gb_model.predict(X_test))

    st.subheader("📊 Model Performance:")
    st.write(f"**Linear Regression R²:** {lr_r2:.3f}")
    st.write(f"**Gradient Boosting R²:** {gb_r2:.3f}")
    st.write(f"**Gradient Boosting MSE:** {mean_squared_error(y_test, gb_model.predict(X_test)):.3f}")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("© 2025 Pratik Banarse | Machine Learning Streamlit App")

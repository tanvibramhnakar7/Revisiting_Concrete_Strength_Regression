🏗️ Concrete Strength Prediction App
📖 Overview:
This project predicts the compressive strength of concrete (csMPa) based on the composition of its ingredients using a Machine Learning regression model.
The app is built with Streamlit, providing an easy-to-use web interface for users to input material quantities and instantly get the predicted concrete strength.

🧠 Objective:
To develop a predictive model that estimates concrete compressive strength using the following input parameters:
FeatureDescriptioncementAmount of cement (kg/m³)slagAmount of blast furnace slag (kg/m³)flyashAmount of fly ash (kg/m³)waterWater content (kg/m³)superplasticizerSuperplasticizer amount (kg/m³)coarseaggregateCoarse aggregate (kg/m³)fineaggregateFine aggregate (kg/m³)ageAge of concrete (days)
Target Variable: csMPa (Concrete compressive strength in MPa)

🧩 Dataset:
Dataset Name: Concrete_Data_Yeh.csv
Source: UCI Machine Learning Repository (Concrete Compressive Strength Dataset)
Shape: 1030 rows × 9 columns

🧰 Technologies Used:
Python, Pandas, NumPy, Scikit-learn, Streamlit

⚙️ Installation and Setup:
1️⃣ Clone the repository
git clone https://github.com/tanvibramhnakar7/concrete-strength-predictor.git
cd concrete-strength-predictor

2️⃣ Install dependencies:
Create a virtual environment (optional) and install packages:
pip install -r requirements.txt

3️⃣ Train the model (if not available):
Run the training script to generate the concrete_model.pkl file:
python train_model.py

4️⃣ Run the Streamlit app:
streamlit run streamlit_app.py

🧮 Model Details:
Algorithm Used: Linear Regression / Random Forest Regressor (depending on your training script)
Evaluation Metric: R² Score / RMSE
The trained model and scaler are stored in a single file:
pickle.dump((model, scaler), open('concrete_model.pkl', 'wb'))

🖥️ App Features:
✅ Interactive UI built with Streamlit
✅ Predicts compressive strength instantly
✅ Easy-to-use input sliders and fields
✅ Lightweight and deployable anywhere (e.g., Streamlit Cloud, Heroku).

👤 Developer Info:
Name: Tanvi Bramhnakar
📧 Email: tanvibramhnakar18@gmail.com
💻 GitHub: github.com/tanvibramhnakar7
🔗 LinkedIn: linkedin.com/in/tanvi-bramhnakar-4b1285294

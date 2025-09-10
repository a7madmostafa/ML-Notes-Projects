import streamlit as st
import numpy as np
import pandas as pd
import pickle, os

# Page config
st.set_page_config(page_title="Car Price Prediction", layout="centered")

# Title and subtitle
st.title("🚗 Car Price Prediction")
st.markdown("Predict the **price of your car** based on its features!")

# Current Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Data Path
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
# Model Path
MODEL_PATH = os.path.join(BASE_DIR, "models", "ml_model.pkl")


# Load the Data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# Load the Model
model = pickle.load(open(MODEL_PATH, "rb"))

# Create two columns
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("📍 Location", df["Location"].unique())
    year = st.slider("📅 Year", 1995, 2019, 2015)
    km = st.slider("🛣️ Kilometers Driven", 150, 300000, 50000)
    brand = st.selectbox("🏷️ Brand", df["Brand"].unique())
    df_brand = df[df["Brand"] == brand]
    model_name = st.selectbox("🚘 Model", df_brand["Model"].unique())
    seats = st.slider("👥 Seats", 2, 8, 4)

with col2:
    df_model = df[df["Model"] == model_name]
    fuel = st.selectbox("⛽ Fuel Type", df_model["Fuel_Type"].unique())
    transmission = st.selectbox("⚙️ Transmission", df_model["Transmission"].unique())
    owner = st.selectbox("👤 Owner Type", df["Owner_Type"].unique())
    mileage = st.slider("📏 Mileage (km/l)", df_model["Mileage"].min(), 500.0, 25.0)
    engine = st.slider("🔧 Engine (CC)", 500, 6000, 1500)
    power = st.slider("💪 Power (bhp)", 50, 500, 100)

# Collect Features
features = pd.DataFrame(
    {
        "Location": location,
        "Kilometers_Driven": km,
        "Fuel_Type": fuel,
        "Transmission": transmission,
        "Owner_Type": owner,
        "Mileage": mileage,
        "Engine": engine,
        "Power": power,
        "Seats": seats,
        "Brand": brand,
        "Model": model_name,
        "Age": df["Year"].max() + 1 - year
    },
    index=[0]
)


# Predict button
st.markdown("---")
if st.button("🔮 Predict Price"):
    prediction = model.predict(features)
    prediction = np.expm1(prediction) * 1200
    st.success(f"💵 Predicted Price: **$ {prediction[0]:,.2f}**")

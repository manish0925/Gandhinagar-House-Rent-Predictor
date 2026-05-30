import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Gandhinagar Rent Predictor", layout="centered")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    pipeline = joblib.load("pipeline.pkl")
    return model, pipeline

try:
    model, pipeline = load_assets()
except Exception as e:
    st.error("Model or Pipeline files not found. Please run your training script first.")
    st.stop()

# --- GUI DESIGN ---
st.title("🏠 Gandhinagar House Rent Predictor")
st.markdown("Enter the details of the property to estimate the monthly rent.")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    bhk = st.selectbox("BHK Type", ["1BHK", "2BHK", "3BHK", "4BHK", "RK"])
    prop_type = st.selectbox("Property Type", ["House", "Apartment", "Tenament", "Penthouse"])
    sharing = st.selectbox("Sharing Type", ["Full", "Shared"])
    gender = st.selectbox("Gender Preference", ["Girls", "Boys", "Any"])
    location = st.selectbox("Location", ["Sector 6", "Raysan", "Sargasan", "Kudasan", "Infocity"])

with col2:
    furnished = st.selectbox("Furnishing Status", ["Basic", "Semi", "Full"])
    ac = st.radio("AC Available?", ["Yes", "No"], horizontal=True)
    fridge = st.radio("Fridge Available?", ["Yes", "No"], horizontal=True)
    washing_machine = st.radio("Washing Machine?", ["Yes", "No"], horizontal=True)
    geyser = st.radio("Geyser Available?", ["Yes", "No"], horizontal=True)

# Additional Features in an expander
with st.expander("More Features"):
    wifi = st.checkbox("WiFi Included", value=True)
    water_24h = st.checkbox("24h Water Supply", value=True)
    parking = st.checkbox("Parking Available")
    cooking = st.checkbox("Cooking Allowed", value=True)
    light_bill = st.checkbox("Light Bill Included")

# --- PREDICTION LOGIC ---
if st.button("Predict Rent", type="primary", use_container_width=True):
    # 1. Create a DataFrame for the input
    # Note: 'area' is omitted because your code drops it before training
    input_dict = {
        "bhk": bhk,
        "property_type": prop_type,
        "sharing_type": sharing,
        "gender_preference": gender,
        "location": location,
        "furnished": furnished,
        "ac": ac,
        "fridge": fridge,
        "washing_machine": washing_machine,
        "geyser": geyser,
        "wifi": "Yes" if wifi else "No",
        "water_24h": "Yes" if water_24h else "No",
        "parking": "Yes" if parking else "No",
        "cooking_allowed": "Yes" if cooking else "No",
        "light_bill_included": "Yes" if light_bill else "No"
    }
    
    input_df = pd.DataFrame([input_dict])

    # 2. Match the data types to the training script
    # Your training script used pd.to_numeric(errors='ignore')
    for col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors="ignore")

    # 3. Transform using Pipeline & Predict
    try:
        prepared_data = pipeline.transform(input_df)
        prediction = model.predict(prepared_data)
        
        # 4. Display Result
        st.success(f"### Estimated Rent: ₹{round(prediction[0], 2)}")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Check if your input features match the columns in your training CSV.")
import streamlit as st
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# Page setup
st.set_page_config(page_title="AgroTech Greenhouse Predictor", page_icon="🌿")

st.title("🌿 Greenhouse Climate Predictor")
st.info("ML for Greenhouse Parameter Prediction (LSTM Model)")

# Load model and scalers
@st.cache_resource
def load_ml_assets():
    model = load_model('lstm_greenhouse_model.h5', compile=False)
    sc_x = joblib.load('scaler_x.pkl')
    sc_y = joblib.load('scaler_y.pkl')
        # Temporary debug line to see the feature names
    return model, sc_x, sc_y



try:
    lstm_model, scaler_x, scaler_y = load_ml_assets()
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")

# User Input Section
st.subheader("Current Sensor Readings")
col1, col2 = st.columns(2)

with col1:
    ec = st.number_input("EC (Electrical Conductivity)", value=1.5)
    tds = st.number_input("TDS (Total Dissolved Solids)", value=700.0)
    turbidity = st.number_input("Turbidity", value=5.0)
    light_level = st.number_input("Light Level (Lux)", value=300.0)

with col2:
    # Adding placeholders for the other 4 features to reach your 8-feature count
    curr_temp = st.number_input("Current Air Temp (°C)", value=22.0)
    curr_hum = st.number_input("Current Humidity (%)", value=50.0)
    curr_co2 = st.number_input("Current CO2 (ppm)", value=400.0)

# Add this temporary line right before the line that causes the error:
st.write(f"DEBUG: Scaler expects {scaler_x.n_features_in_} features.")
if hasattr(scaler_x, "feature_names_in_"):
    st.write("### 📋 Features your model expects:")
    st.write(scaler_x.feature_names_in_)
else:
    st.write("Scaler doesn't have names, but it wants 10 values.")
        
    
if st.button("Generate 1-Minute Prediction"):
    # THE CRITICAL STEP: Mapping the sliders to the 10 features in order
    # Order from your image: ec, tds, turbidity, light_level, 
    # air_temp_lag1, air_temp_lag5, hum_lag1, hum_lag5, co2_lag1, co2_lag5
    
    input_row = [
        ec,              # 1. ec
        tds,             # 2. tds
        turbidity,       # 3. turbidity
        light_level,     # 4. light_level
        curr_temp,       # 5. air_temperature_lag1
        curr_temp,       # 6. air_temperature_lag5
        curr_hum,        # 7. air_humidity_lag1
        curr_hum,        # 8. air_humidity_lag5
        curr_co2,        # 9. co2_lag1
        curr_co2         # 10. co2_lag5
    ]
    
    # Verify the count one last time
    if len(input_row) == 10:
        # Create sequence for LSTM (10 time steps)
        sequence = np.tile(input_row, (10, 1)) 
        
        # Scale and Predict
        scaled_seq = scaler_x.transform(sequence)
        input_ready = np.expand_dims(scaled_seq, axis=0)
        
        pred_scaled = lstm_model.predict(input_ready)
        prediction = scaler_y.inverse_transform(pred_scaled)
        
        # Display Results
        st.divider()
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Predicted Temp", f"{prediction[0][0]:.2f} °C")
        res_col2.metric("Predicted Humidity", f"{prediction[0][1]:.2f} %")
        res_col3.metric("Predicted CO2", f"{prediction[0][2]:.2f} ppm")
    else:
        st.error("Feature count mismatch! Check the input_row logic.")

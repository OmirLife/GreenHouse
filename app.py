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
    temp = st.number_input("Air Temperature (°C)", value=22.0)
    hum = st.number_input("Humidity (%)", value=50.0)
    light = st.number_input("Light Level (Lux)", value=300.0)
    ext_temp = st.number_input("External Temp (°C)", value=15.0)

with col2:
    # Adding placeholders for the other 4 features to reach your 8-feature count
    feat5 = st.number_input("Feature 5", value=0.0)
    feat6 = st.number_input("Feature 6", value=0.0)
    feat7 = st.number_input("Feature 7", value=0.0)
    feat8 = st.number_input("Feature 8", value=0.0)

# Add this temporary line right before the line that causes the error:
st.write(f"DEBUG: Scaler expects {scaler_x.n_features_in_} features.")
if hasattr(sc_x, "feature_names_in_"):
        st.write("### 📋 Features your model expects:")
        st.write(sc_x.feature_names_in_)
    else:
        st.write("Scaler doesn't have names, but it wants 10 values.")
        
if st.button("Generate 1-Minute Prediction"):
    # LSTM needs a sequence of 10. We'll duplicate the current state 
    # to simulate a stable greenhouse for this demo.
    input_row = [temp, hum, light, ext_temp, feat5, feat6, feat7, feat8]
    sequence = np.tile(input_row, (10, 1)) 
    
    # Scaling and Prediction
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

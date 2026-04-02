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

st.subheader("📊 Model Input Parameters")
st.info("Adjust the current state and historical lags to see how the LSTM responds.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Water & Light**")
    val_ec = st.number_input("EC (mS/cm)", value=1.5)
    val_tds = st.number_input("TDS (ppm)", value=700.0)
    val_turb = st.number_input("Turbidity (NTU)", value=5.0)
    val_light = st.number_input("Light Level (Lux)", value=300.0)

with col2:
    st.markdown("**Temperature Lags**")
    t_lag1 = st.number_input("Air Temp Lag 1 (min)", value=22.0)
    t_lag5 = st.number_input("Air Temp Lag 5 (min)", value=21.5)
    
    st.markdown("**Humidity Lags**")
    h_lag1 = st.number_input("Humidity Lag 1 (min)", value=50.0)
    h_lag5 = st.number_input("Humidity Lag 5 (min)", value=48.0)

with col3:
    st.markdown("**CO2 Lags**")
    c_lag1 = st.number_input("CO2 Lag 1 (min)", value=400.0)
    c_lag5 = st.number_input("CO2 Lag 5 (min)", value=395.0)

# The Button Logic now maps 1:1 to the 10 variables
if st.button("Generate 1-Minute Prediction"):
    input_row = [
        val_ec, val_tds, val_turb, val_light, # 1-4
        t_lag1, t_lag5,                      # 5-6
        h_lag1, h_lag5,                      # 7-8
        c_lag1, c_lag5                       # 9-10
    ]

# # Add this temporary line right before the line that causes the error:
# st.write(f"DEBUG: Scaler expects {scaler_x.n_features_in_} features.")
# if hasattr(scaler_x, "feature_names_in_"):
#     st.write("### 📋 Features your model expects:")
#     st.write(scaler_x.feature_names_in_)
# else:
#     st.write("Scaler doesn't have names, but it wants 10 values.")
        
    

    
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

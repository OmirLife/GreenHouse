import streamlit as st
import numpy as np
import joblib
import time
import pandas as pd
import os
from tensorflow.keras.models import load_model

# Page setup
st.set_page_config(page_title="AgroTech Real-Time Monitor", page_icon="🌿", layout="wide")

st.title("🌿 Greenhouse Digital Twin: Live Monitoring")
st.markdown("Automated LSTM Prediction Pipeline (1-Minute Resolution)")

# --- CONTAINERS FOR UPDATING UI ---
metric_row = st.empty()
chart_container = st.empty()
status_sidebar = st.sidebar.empty()

# --- ASSET LOADING ---
@st.cache_resource
def load_ml_assets():
    try:
        model = load_model('lstm_greenhouse_model.h5', compile=False)
        sc_x = joblib.load('scaler_x.pkl')
        sc_y = joblib.load('scaler_y.pkl')
        return model, sc_x, sc_y
    except Exception as e:
        return None, None, str(e)

lstm_model, scaler_x, scaler_y = load_ml_assets()

# --- LIVE LOOP ---
FILE_PATH = "live_greenhouse_data.csv"

while True:
    if os.path.exists(FILE_PATH):
        try:
            df = pd.read_csv(FILE_PATH)
            
            # We need at least 6 rows to get the 5-minute lag
            if len(df) >= 6:
                # 1. GET LATEST SENSOR READINGS
                latest = df.iloc[-1]
                prev_1min = df.iloc[-2]
                prev_5min = df.iloc[-6]

                # 2. CONSTRUCT FEATURE VECTOR (The 10 "Plugs")
                # Order must match your model's requirement exactly
                input_row = [
                    latest['ec'], latest['tds'], latest['turbidity'], latest['light_level'],
                    prev_1min['air_temperature'], prev_5min['air_temperature'], # temp lags
                    prev_1min['air_humidity'],    prev_5min['air_humidity'],    # hum lags
                    prev_1min['co2'],             prev_5min['co2']              # co2 lags
                ]

                # 3. RUN PREDICTION
                sequence = np.tile(input_row, (10, 1))
                scaled_seq = scaler_x.transform(sequence)
                input_ready = np.expand_dims(scaled_seq, axis=0)
                
                pred_scaled = lstm_model.predict(input_ready, verbose=0)
                prediction = scaler_y.inverse_transform(pred_scaled)

                # 4. UPDATE METRICS
                with metric_row.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Air Temp", f"{latest['air_temperature']:.2f} °C")
                    c2.metric("Humidity", f"{latest['air_humidity']:.1f} %")
                    c3.metric("CO2 Level", f"{latest['co2']:.0f} ppm")
                    c4.metric("EC (Nutrients)", f"{latest['ec']:.2f}")

                # 5. UPDATE CHARTS (Visualizing History)
                with chart_container.container():
                    st.divider()
                    st.subheader("📈 Historical Trends & Predictions")
                    # Show last 50 minutes of data
                    history_view = df.tail(50).set_index("timestamp")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.line_chart(history_view[["air_temperature", "air_humidity"]])
                    with col_b:
                        st.line_chart(history_view["co2"])
                    
                    st.success(f"Forecast for next minute: Temp {prediction[0][0]:.2f}°C | CO2 {prediction[0][2]:.0f}ppm")

                status_sidebar.write(f"🔄 Last Update: {latest['timestamp']}")
            else:
                status_sidebar.warning(f"⏳ Gathering data... ({len(df)}/6 rows)")
        
        except Exception as e:
            status_sidebar.error(f"❌ Read Error: {e}")
    else:
        status_sidebar.info("📡 Waiting for Simulator to start...")

    time.sleep(5) # Refresh rate (5 seconds is good for demo)

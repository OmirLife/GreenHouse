import pandas as pd
import numpy as np
import time
from datetime import datetime
import os

def run_simulator():
    print("🚀 Greenhouse Simulator Started. Sending data to live_greenhouse_data.csv...")
    
    # Starting values
    temp, hum, co2 = 22.0, 50.0, 400.0
    
    while True:
        now = datetime.now().strftime("%H:%M:%S")
        
        # Simulate slight variations + a 24h sine wave pattern
        t_val = temp + np.random.normal(0, 0.1)
        h_val = hum + np.random.normal(0, 0.2)
        c_val = co2 + np.random.normal(0, 5)
        
        new_row = {
            "timestamp": now,
            "ec": 1.5 + np.random.normal(0, 0.02),
            "tds": 700 + np.random.normal(0, 5),
            "turbidity": 5.0 + np.random.normal(0, 0.1),
            "light_level": 300 + 50 * np.sin(time.time()/100),
            "air_temperature": t_val,
            "air_humidity": h_val,
            "co2": c_val
        }
        
        df_new = pd.DataFrame([new_row])
        # Append to CSV
        df_new.to_csv("live_greenhouse_data.csv", mode='a', index=False, header=not os.path.exists("live_greenhouse_data.csv"))
        
        print(f"[{now}] Data point sent.")
        time.sleep(2) # Speeding it up for the demo (Real time would be 60s)

if __name__ == "__main__":
    run_simulator()

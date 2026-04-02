import pandas as pd
import numpy as np
import time
from datetime import datetime

def generate_greenhouse_data():
    while True:
        # 1. Create realistic values with slight variations
        now = datetime.now()
        second = now.second
        
        # Simulating a day/night cycle using a sine wave
        temp = 22 + 5 * np.sin(second / 10) + np.random.normal(0, 0.1)
        hum = 50 + 10 * np.cos(second / 10) + np.random.normal(0, 0.5)
        co2 = 400 + 50 * np.sin(second / 15) + np.random.normal(0, 5)
        
        new_data = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "ec": 1.5 + np.random.normal(0, 0.05),
            "tds": 700 + np.random.normal(0, 10),
            "turbidity": 5.0 + np.random.normal(0, 0.2),
            "light_level": 300 + 200 * np.sin(second / 10),
            "air_temperature": temp,
            "air_humidity": hum,
            "co2": co2
        }
        
        # 2. Append to a CSV file
        df = pd.DataFrame([new_data])
        df.to_csv("live_greenhouse_data.csv", mode='a', index=False, header=not os.path.exists("live_greenhouse_data.csv"))
        
        print(f"Sent to Greenhouse Data Lake: {now.strftime('%H:%M:%S')}")
        time.sleep(5) # New data every 5 seconds

if __name__ == "__main__":
    import os
    generate_greenhouse_data()

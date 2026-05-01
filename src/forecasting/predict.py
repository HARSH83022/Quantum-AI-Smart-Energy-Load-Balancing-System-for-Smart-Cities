# import pandas as pd
# import numpy as np
# import joblib
# import holidays


# class Predictor:

#     def __init__(self):

#         # Load trained models
#         self.load_model = joblib.load("models/peak_load_model.pkl")
#         self.time_model = joblib.load("models/peak_time_model.pkl")

#         # Load dataset
#         self.df = pd.read_excel("data/raw/preeeeeee.xlsx")

#         # Rename column
#         self.df = self.df.rename(columns={"DATES": "DATE"})

#         # Convert DATE
#         self.df["DATE"] = pd.to_datetime(self.df["DATE"], dayfirst=True)

#         # Convert PEAK_TIME → numeric hours
#         self.df["PEAK_TIME"] = pd.to_datetime(
#             self.df["PEAK_TIME"].astype(str), format="%H:%M:%S"
#         )

#         self.df["PEAK_TIME"] = (
#             self.df["PEAK_TIME"].dt.hour
#             + self.df["PEAK_TIME"].dt.minute / 60
#             + self.df["PEAK_TIME"].dt.second / 3600
#         )

#         # Sort
#         self.df = self.df.sort_values("DATE").reset_index(drop=True)

#         # 🔥 AUTO INDIA HOLIDAYS
#         self.india_holidays = holidays.India()


#     # 🔥 NEW HOLIDAY FUNCTION
#     def is_holiday(self, date):

#         return 1 if date in self.india_holidays else 0


#     def predict_from_date(self, date):

#         try:

#             date = pd.to_datetime(date)

#             history = self.df[self.df["DATE"] < date]

#             if len(history) < 7:
#                 return {"error": "Not enough historical data"}

#             # ---------------------------
#             # Feature Engineering
#             # ---------------------------

#             last_row = history.iloc[-1]

#             lag_1 = last_row["PEAK_VALUE_MW"]
#             lag_7 = history.iloc[-7]["PEAK_VALUE_MW"]

#             rolling_3 = history.tail(3)["PEAK_VALUE_MW"].mean()
#             rolling_7 = history.tail(7)["PEAK_VALUE_MW"].mean()

#             lag_peak_time = last_row["PEAK_TIME"]

#             day_of_week = date.dayofweek
#             month = date.month

#             is_weekend = 1 if day_of_week >= 5 else 0
#             is_holiday = self.is_holiday(date)

#             # ---------------------------
#             # Feature Vector
#             # ---------------------------

#             X = np.array([[
#                 day_of_week,
#                 month,
#                 is_weekend,
#                 is_holiday,
#                 lag_1,
#                 lag_7,
#                 rolling_3,
#                 rolling_7,
#                 lag_peak_time
#             ]])

#             # ---------------------------
#             # Prediction
#             # ---------------------------

#             pred_load = float(self.load_model.predict(X)[0])
#             pred_time = float(self.time_model.predict(X)[0])

#             # ---------------------------
#             # Transformer Logic
#             # ---------------------------

#             if pred_load > 6500:
#                 transformer_status = "Overloaded ⚠"
#             elif pred_load < 4500:
#                 transformer_status = "Underutilized"
#             else:
#                 transformer_status = "Normal"

#             # ---------------------------
#             # Output
#             # ---------------------------

#             return {
#                 "predicted_peak_load_MW": round(pred_load, 2),
#                 "predicted_peak_hour": round(pred_time, 2),
#                 "is_holiday": int(is_holiday),
#                 "is_weekend": int(is_weekend),
#                 "day_name": date.strftime("%A"),
#                 "transformer_status": transformer_status
#             }

#         except Exception as e:
#             return {"error": str(e)}


import pandas as pd
import numpy as np
import joblib
import holidays
import requests


class Predictor:

    def __init__(self):

        self.load_model = joblib.load("models/peak_load_model.pkl")
        self.time_model = joblib.load("models/peak_time_model.pkl")

        self.df = pd.read_excel("data/raw/preeeeeee.xlsx")

        self.df = self.df.rename(columns={"DATES": "DATE"})
        self.df["DATE"] = pd.to_datetime(self.df["DATE"], dayfirst=True)

        self.df["PEAK_TIME"] = pd.to_datetime(
            self.df["PEAK_TIME"].astype(str), format="%H:%M:%S"
        )

        self.df["PEAK_TIME"] = (
            self.df["PEAK_TIME"].dt.hour +
            self.df["PEAK_TIME"].dt.minute / 60 +
            self.df["PEAK_TIME"].dt.second / 3600
        )

        self.df = self.df.sort_values("DATE").reset_index(drop=True)

        self.india_holidays = holidays.India()
        
        # Cache for weather data by date
        self.weather_cache = {}

    # 🌡 Smart Temperature from Open-Meteo API (with caching)
    def get_temperature(self, date):
        date_str = date.strftime("%Y-%m-%d")
        
        # Check cache first
        if date_str in self.weather_cache:
            return self.weather_cache[date_str]['temperature']
        
        try:
            # Delhi coordinates
            lat, lon = 28.7041, 77.1025
            
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&hourly=temperature_2m"
            response = requests.get(url)
            data = response.json()
            
            temps = data['hourly']['temperature_2m']
            temp = round(np.mean(temps), 1)  # Average temperature for the day
            
            # Cache the result
            if date_str not in self.weather_cache:
                self.weather_cache[date_str] = {}
            self.weather_cache[date_str]['temperature'] = temp
            
            return temp
            
        except Exception as e:
            print(f"API failed: {e}, using fallback")
            # Fallback to old logic
            month = date.month
            if month in [12,1,2]:
                base = np.random.uniform(8,16)
            elif month in [4,5,6]:
                base = np.random.uniform(35,44)
            elif month in [7,8,9]:
                base = np.random.uniform(25,32)
            else:
                base = np.random.uniform(18,30)
            temp = round(base,1)
            
            # Cache fallback result too
            if date_str not in self.weather_cache:
                self.weather_cache[date_str] = {}
            self.weather_cache[date_str]['temperature'] = temp
            
            return temp

    # 🌧 Rain + humidity from Open-Meteo API (with caching)
    def get_weather_factors(self, date):
        date_str = date.strftime("%Y-%m-%d")
        
        # Check cache first
        if date_str in self.weather_cache and 'rain' in self.weather_cache[date_str]:
            rain = self.weather_cache[date_str]['rain']
            humidity = self.weather_cache[date_str]['humidity']
            return rain, humidity
        
        try:
            # Delhi coordinates
            lat, lon = 28.7041, 77.1025
            
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&hourly=relative_humidity_2m,precipitation"
            response = requests.get(url)
            data = response.json()
            
            humidities = data['hourly']['relative_humidity_2m']
            precipitations = data['hourly']['precipitation']
            
            rain = round(np.sum(precipitations), 2)  # Total precipitation for the day
            humidity = round(np.mean(humidities), 1)  # Average humidity
            
            # Cache the results
            if date_str not in self.weather_cache:
                self.weather_cache[date_str] = {}
            self.weather_cache[date_str]['rain'] = rain
            self.weather_cache[date_str]['humidity'] = humidity
            
            return rain, humidity
            
        except Exception as e:
            print(f"API failed: {e}, using fallback")
            # Fallback to old logic
            month = date.month
            if month in [7,8]:
                rain = np.random.uniform(0.5,1.0)
                humidity = np.random.uniform(70,90)
            else:
                rain = np.random.uniform(0.0,0.3)
                humidity = np.random.uniform(40,70)
            
            # Cache fallback results too
            if date_str not in self.weather_cache:
                self.weather_cache[date_str] = {}
            self.weather_cache[date_str]['rain'] = rain
            self.weather_cache[date_str]['humidity'] = humidity
            
            return rain, humidity

    # 🌤 Weather effect
    def adjust_load(self, load, temp, rain, humidity):

        if temp > 38:
            load *= 1.08

        if rain > 0.5:
            load *= 0.95

        if humidity > 75:
            load *= 1.03

        return load

    # ⚡ Outage risk
    def outage_risk(self, load, temp):

        score = 0

        if load > 7000: score += 50
        if temp > 40: score += 30
        if load > 6500: score += 20

        if score >= 70: return "High Risk 🚨"
        elif score >= 40: return "Moderate ⚠"
        else: return "Low Risk ✅"

    def is_holiday(self, date):
        return 1 if date in self.india_holidays else 0

    def predict_from_date(self, date):

        try:
            date = pd.to_datetime(date)

            history = self.df[self.df["DATE"] < date]

            if len(history) < 7:
                return {"error": "Not enough data"}

            last = history.iloc[-1]

            lag_1 = last["PEAK_VALUE_MW"]
            lag_7 = history.iloc[-7]["PEAK_VALUE_MW"]

            rolling_3 = history.tail(3)["PEAK_VALUE_MW"].mean()
            rolling_7 = history.tail(7)["PEAK_VALUE_MW"].mean()

            lag_time = last["PEAK_TIME"]

            day = date.dayofweek
            month = date.month

            is_weekend = 1 if day >= 5 else 0
            is_holiday = self.is_holiday(date)

            # 🌡 Weather
            temp = self.get_temperature(date)
            rain, humidity = self.get_weather_factors(date)

            X = np.array([[
                day, month, is_weekend, is_holiday,
                lag_1, lag_7, rolling_3, rolling_7, lag_time
            ]])

            load = float(self.load_model.predict(X)[0])
            peak_time = float(self.time_model.predict(X)[0])

            # Apply intelligence
            load = self.adjust_load(load, temp, rain, humidity)

            # Transformer
            if load > 7000:
                transformer = "Critical 🔴"
            elif load > 6000:
                transformer = "Overloaded ⚠"
            elif load < 4000:
                transformer = "Underutilized 🟡"
            else:
                transformer = "Optimal 🟢"

            return {
                "predicted_peak_load_MW": round(load,2),
                "predicted_peak_hour": round(peak_time,2),
                "temperature": temp,
                "rain": round(rain,2),
                "humidity": int(humidity),
                "is_holiday": is_holiday,
                "is_weekend": is_weekend,
                "day_name": date.strftime("%A"),
                "transformer_status": transformer,
                "outage_risk": self.outage_risk(load, temp)
            }

        except Exception as e:
            return {"error": str(e)}
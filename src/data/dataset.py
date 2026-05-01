# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler


# class SmartGridDataset:

#     def __init__(self):

#         self.data_path = "data/processed/clean_delhi_smart_grid_dataset.csv"

#         self.sequence_length = 24

#         self.scaler = MinMaxScaler()

#         self.features = [
#             "demand_MW",
#             "temperature",
#             "humidity",
#             "hour_sin",
#             "hour_cos",
#             "day_sin",
#             "day_cos",
#             "month",
#             "is_weekend"
#         ]


#     def load_dataset(self):

#         print("Loading dataset...")

#         df = pd.read_csv(self.data_path)

#         df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)

#         df = df.sort_values("timestamp")

#         df["hour"] = df["timestamp"].dt.hour
#         df["day_of_week"] = df["timestamp"].dt.dayofweek
#         df["month"] = df["timestamp"].dt.month

#         # ✅ Cyclical encoding (correct indentation)
#         df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
#         df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

#         df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
#         df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

#         df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

#         df = df[self.features]

#         data = df.values.astype(np.float32)

#         scaled = self.scaler.fit_transform(data)

#         return scaled


#     def create_sequences(self, data):

#         X = []
#         y = []

#         for i in range(len(data) - self.sequence_length):
#             X.append(data[i:i+self.sequence_length])
#             y.append(data[i+self.sequence_length][0])

#         X = np.array(X, dtype=np.float32)
#         y = np.array(y, dtype=np.float32)

#         return X, y


#     def prepare_data(self):

#         data = self.load_dataset()

#         X, y = self.create_sequences(data)

#         # ✅ Random sampling for faster training
#         max_samples = 200000

#         if len(X) > max_samples:
#             indices = np.random.choice(len(X), max_samples, replace=False)
#             X = X[indices]
#             y = y[indices]

#         return X, y, self.scaler


# if __name__ == "__main__":

#     dataset = SmartGridDataset()

#     X, y, scaler = dataset.prepare_data()

#     print("Dataset prepared successfully")
#     print("X shape:", X.shape)
#     print("y shape:", y.shape)





import pandas as pd


class SmartGridDataset:

    def __init__(self):

        # real peak dataset
        self.peak_path = "data/raw/preeeeeee.xlsx"

        # holiday dataset
        self.holiday_path = "data/processed/extended_holidays_2025_2026.csv"


    def load_peak_data(self):

        print("Loading peak dataset...")

        df = pd.read_excel(self.peak_path)

        df["DATES"] = pd.to_datetime(df["DATES"])

        # convert peak time to numeric hour
        df["PEAK_TIME"] = pd.to_datetime(df["PEAK_TIME"], format="%H:%M:%S")

        df["peak_hour"] = (
            df["PEAK_TIME"].dt.hour +
            df["PEAK_TIME"].dt.minute / 60 +
            df["PEAK_TIME"].dt.second / 3600
        )

        return df


    def load_holiday_data(self):

        holidays = pd.read_csv(self.holiday_path)

        holidays["date"] = pd.to_datetime(holidays["date"])

        return holidays


    def merge_datasets(self):

        peak = self.load_peak_data()
        holidays = self.load_holiday_data()

        peak = peak.rename(columns={"DATES": "date"})

        df = peak.merge(holidays, on="date", how="left")

        df["is_holiday"] = df["is_holiday"].fillna(0)

        return df


    def create_features(self):

        df = self.merge_datasets()

        # date features
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month

        # lag features
        df["lag_1"] = df["PEAK_VALUE_MW"].shift(1)
        df["lag_7"] = df["PEAK_VALUE_MW"].shift(7)

        # rolling averages
        df["rolling_3"] = df["PEAK_VALUE_MW"].rolling(3).mean()
        df["rolling_7"] = df["PEAK_VALUE_MW"].rolling(7).mean()

        df = df.dropna()

        return df


    def get_training_data(self):

        df = self.create_features()

        features = [
            "day_of_week",
            "month",
            "is_weekend",
            "is_holiday",
            "lag_1",
            "lag_7",
            "rolling_3",
            "rolling_7"
        ]

        X = df[features]

        y_load = df["PEAK_VALUE_MW"]
        y_time = df["peak_hour"]

        return X, y_load, y_time


if __name__ == "__main__":

    dataset = SmartGridDataset()

    X, y_load, y_time = dataset.get_training_data()

    print("Dataset prepared successfully")

    print("Feature shape:", X.shape)
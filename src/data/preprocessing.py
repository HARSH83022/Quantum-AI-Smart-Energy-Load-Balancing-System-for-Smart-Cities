import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataPreprocessor:

    def __init__(self):
        self.file_path = "data/raw/preeeeeee.xlsx"

    def load_data(self):

        print("Loading peak dataset...")

        df = pd.read_excel(self.file_path)

        print("Columns found:", df.columns)

        # Rename DATE column
        df = df.rename(columns={"DATES": "DATE"})

        df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True)

        df = df.sort_values("DATE").reset_index(drop=True)

        return df

    def convert_time_to_hours(self, time_str):

        # Convert HH:MM:SS → decimal hours
        t = pd.to_datetime(time_str, format="%H:%M:%S")

        return t.hour + t.minute / 60 + t.second / 3600

    def create_features(self, df):

        # ---------------------------
        # Convert PEAK_TIME → numeric
        # ---------------------------
        df["PEAK_TIME"] = df["PEAK_TIME"].astype(str)

        df["PEAK_TIME"] = df["PEAK_TIME"].apply(self.convert_time_to_hours)

        # ---------------------------
        # Calendar Features
        # ---------------------------
        df["day_of_week"] = df["DATE"].dt.dayofweek
        df["month"] = df["DATE"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        df["is_holiday"] = 0

        # ---------------------------
        # Load Features
        # ---------------------------
        df["lag_1"] = df["PEAK_VALUE_MW"].shift(1)
        df["lag_7"] = df["PEAK_VALUE_MW"].shift(7)

        df["rolling_3"] = df["PEAK_VALUE_MW"].rolling(3).mean()
        df["rolling_7"] = df["PEAK_VALUE_MW"].rolling(7).mean()

        # ---------------------------
        # 🔥 Peak Time Feature (IMPORTANT)
        # ---------------------------
        df["lag_peak_time"] = df["PEAK_TIME"].shift(1)

        # ---------------------------
        # Drop NA
        # ---------------------------
        df = df.dropna().reset_index(drop=True)

        return df

    def prepare_training_data(self):

        print("Preparing training data...")

        df = self.load_data()

        df = self.create_features(df)

        feature_cols = [
            "day_of_week",
            "month",
            "is_weekend",
            "is_holiday",
            "lag_1",
            "lag_7",
            "rolling_3",
            "rolling_7",
            "lag_peak_time"
        ]

        X = df[feature_cols].values

        y_load = df["PEAK_VALUE_MW"].values
        y_time = df["PEAK_TIME"].values

        # Time-series split
        X_train, X_test, y_load_train, y_load_test = train_test_split(
            X, y_load, test_size=0.2, shuffle=False
        )

        _, _, y_time_train, y_time_test = train_test_split(
            X, y_time, test_size=0.2, shuffle=False
        )

        print("Training samples:", len(X_train))
        print("Testing samples:", len(X_test))

        return (
            X_train,
            X_test,
            y_load_train,
            y_load_test,
            y_time_train,
            y_time_test
        )


if __name__ == "__main__":

    pre = DataPreprocessor()

    X_train, X_test, y_load_train, y_load_test, y_time_train, y_time_test = pre.prepare_training_data()

    print("X_train shape:", X_train.shape)
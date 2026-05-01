# import numpy as np
# import joblib

# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# from src.data.preprocessing import DataPreprocessor


# class PeakLoadModel:

#     def __init__(self):

#         self.preprocessor = DataPreprocessor()


#     def train(self):

#         (
#             X_train,
#             X_test,
#             y_load_train,
#             y_load_test,
#             y_time_train,
#             y_time_test
#         ) = self.preprocessor.prepare_training_data()

#         print("\n==============================")
#         print("Training Peak Load Model")
#         print("==============================")

#         load_model = XGBRegressor(
#             n_estimators=300,
#             max_depth=5,
#             learning_rate=0.05,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42
#         )

#         load_model.fit(X_train, y_load_train)

#         preds = load_model.predict(X_test)

#         mae = mean_absolute_error(y_load_test, preds)

#         rmse = np.sqrt(mean_squared_error(y_load_test, preds))

#         mape = np.mean(np.abs((y_load_test - preds) / y_load_test)) * 100

#         accuracy = 100 - mape

#         print("\nPeak Load Model Performance")
#         print("---------------------------")
#         print("MAE  :", round(mae, 2), "MW")
#         print("RMSE :", round(rmse, 2), "MW")
#         print("MAPE :", round(mape, 2), "%")
#         print("Accuracy :", round(accuracy, 2), "%")


#         print("\n==============================")
#         print("Training Peak Time Model")
#         print("==============================")

#         time_model = XGBRegressor(
#             n_estimators=200,
#             max_depth=4,
#             learning_rate=0.05,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42
#         )

#         time_model.fit(X_train, y_time_train)

#         preds_time = time_model.predict(X_test)

#         mae_time = mean_absolute_error(y_time_test, preds_time)

#         rmse_time = np.sqrt(mean_squared_error(y_time_test, preds_time))

#         print("\nPeak Time Model Performance")
#         print("---------------------------")
#         print("MAE (hours)  :", round(mae_time, 2))
#         print("RMSE (hours) :", round(rmse_time, 2))


#         # Save models
#         joblib.dump(load_model, "models/peak_load_model.pkl")
#         joblib.dump(time_model, "models/peak_time_model.pkl")

#         print("\nModels saved successfully in /models folder")


# if __name__ == "__main__":

#     trainer = PeakLoadModel()

#     trainer.train()

import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.preprocessing import DataPreprocessor


class PeakLoadModel:

    def __init__(self):
        self.preprocessor = DataPreprocessor()

    def evaluate(self, y_true, preds):

        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))

        # Safe SMAPE (industry preferred)
        smape = np.mean(
            2 * np.abs(preds - y_true) /
            (np.abs(preds) + np.abs(y_true) + 1e-8)
        ) * 100

        accuracy = 100 - smape

        return mae, rmse, smape, accuracy

    def train(self):

        (
            X_train,
            X_test,
            y_load_train,
            y_load_test,
            y_time_train,
            y_time_test
        ) = self.preprocessor.prepare_training_data()

        print("\n==============================")
        print("Training Peak Load Model")
        print("==============================")

        # 🚀 Improved XGBoost
        load_model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )

        load_model.fit(X_train, y_load_train)

        preds = load_model.predict(X_test)

        mae, rmse, smape, accuracy = self.evaluate(y_load_test, preds)

        print("\n📊 Peak Load Model Performance")
        print("--------------------------------")
        print(f"MAE   : {mae:.2f} MW")
        print(f"RMSE  : {rmse:.2f} MW")
        print(f"SMAPE : {smape:.2f}%")
        print(f"🎯 Accuracy ≈ {accuracy:.2f}%")

        # -------------------------------
        # Peak Time Model
        # -------------------------------

        print("\n==============================")
        print("Training Peak Time Model")
        print("==============================")

        time_model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )

        time_model.fit(X_train, y_time_train)

        preds_time = time_model.predict(X_test)

        mae_time, rmse_time, smape_time, acc_time = self.evaluate(
            y_time_test, preds_time
        )

        print("\n📊 Peak Time Model Performance")
        print("--------------------------------")
        print(f"MAE   : {mae_time:.2f} hours")
        print(f"RMSE  : {rmse_time:.2f} hours")
        print(f"SMAPE : {smape_time:.2f}%")
        print(f"🎯 Accuracy ≈ {acc_time:.2f}%")

        # -------------------------------
        # Save Models
        # -------------------------------

        joblib.dump(load_model, "models/peak_load_model.pkl")
        joblib.dump(time_model, "models/peak_time_model.pkl")

        print("\n✅ Models saved successfully in /models folder")

        return load_model, time_model


if __name__ == "__main__":

    trainer = PeakLoadModel()
    trainer.train()
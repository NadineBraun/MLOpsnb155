import mlflow
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from pathlib import Path

def main():
    base_path = Path(__file__).parent
    data_path = base_path / "data"
    model_output = base_path / "model_output"
    model_output.mkdir(exist_ok=True)

    target_col = "trip_count"
    feature_cols = ['hour', 'weekday', 'month', 'year']

    train_path = data_path / "dataset_train.parquet"
    test_path = data_path / "dataset_test.parquet"

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    X_train = df_train[feature_cols].astype("int")
    y_train = df_train[target_col]
    X_test = df_test[feature_cols].astype("int")
    y_test = df_test[target_col]

    model = XGBRegressor(
        n_estimators=167,
        max_depth=13,
        learning_rate=0.03478742290931512,
        subsample=0.7348666779615057,
        colsample_bytree=0.6070931662048404,
        gamma=0.5496341248916392,
        reg_alpha=0.003425043421816794,
        reg_lambda=0.9373031835287268,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mlflow.set_experiment("model_pipeline")
    with mlflow.start_run():
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_param("target", target_col)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        model_path = model_output / "xgboost_model.json"
        model.save_model(str(model_path))
        mlflow.log_artifact(str(model_path))

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }
        metrics_path = model_output / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(str(metrics_path))

        print("✅ Modelltraining abgeschlossen und gespeichert.")

if __name__ == "__main__":
    main()

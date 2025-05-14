import argparse
import mlflow
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    args = parser.parse_args()

    print("Lade Trainings- und Testdaten ...")
    df_train = pd.read_parquet(args.train_path)
    df_test = pd.read_parquet(args.test_path)

    feature_cols = ['hour', 'weekday', 'month', 'year']
    X_train = df_train[feature_cols]
    y_train = df_train[args.target_col]
    X_test = df_test[feature_cols]
    y_test = df_test[args.target_col]

    print("Trainiere Modell mit Optuna-Parametern ...")
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

    print("Bewerte Modell ...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("features", ",".join(feature_cols))
    mlflow.log_param("target", args.target_col)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    reference_path = Path("reference_metrics.json")
    if reference_path.exists():
        with open(reference_path, "r") as f:
            reference_metrics = json.load(f)
        ref_r2 = reference_metrics.get("R2", None)
        if ref_r2 is not None:
            r2_drop = ref_r2 - r2
            mlflow.log_metric("R2_drop", r2_drop)
            if r2_drop > 0.10:
                print(f"⚠️ Warnung: R² ist um {r2_drop:.3f} gefallen im Vergleich zum Referenzwert ({ref_r2:.3f})")
            else:
                print(f"✅ R² liegt im erwarteten Bereich (Abweichung: {r2_drop:.3f})")
        else:
            print("ℹ️ Kein R²-Referenzwert gefunden.")
    else:
        print("ℹ️ Keine Referenzmetrik-Datei gefunden. Monitoring wird übersprungen.")

    model_output = Path("model_output")
    model_output.mkdir(exist_ok=True)
    model_path = model_output / "xgboost_model.json"
    model.save_model(str(model_path))
    mlflow.log_artifact(str(model_path))

    print("Modelltraining abgeschlossen.")

if __name__ == "__main__":
    main()
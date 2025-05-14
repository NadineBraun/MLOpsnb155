import argparse
import mlflow
from pathlib import Path
import pandas as pd
import os
import hashlib

def load_feature_files(years, features_path="features"):
    dfs = []
    for year in years:
        for month in range(1, 13):
            file = Path(features_path) / f"features_yellow_tripdata_{year}-{month:02d}.parquet"
            if file.exists():
                dfs.append(pd.read_parquet(file))
            else:
                print(f"Datei nicht gefunden: {file}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_years", type=str, required=True)
    parser.add_argument("--test_years", type=str, required=True)
    parser.add_argument("--features_path", type=str, default="features")
    parser.add_argument("--output_path", type=str, default=".")
    args = parser.parse_args()

    train_years = [int(y) for y in args.train_years.split(",")]
    test_years = [int(y) for y in args.test_years.split(",")]

    with mlflow.start_run(run_name="split_train_test", nested=True):
        print("Starte Aufteilung in Trainings- und Testdaten ...")
        mlflow.log_param("train_years", args.train_years)
        mlflow.log_param("test_years", args.test_years)

        print("Lade Trainingsdaten ...")
        df_train = load_feature_files(train_years, args.features_path)
        print(f"Trainingsdaten geladen: {df_train.shape}")

        print("Lade Testdaten ...")
        df_test = load_feature_files(test_years, args.features_path)
        print(f"Testdaten geladen: {df_test.shape}")

        if df_train.empty or df_test.empty:
            raise ValueError("❌ Trainings- oder Testdaten fehlen. Aufteilung abgebrochen.")

        train_path = Path(args.output_path) / "df_train.parquet"
        test_path = Path(args.output_path) / "df_test.parquet"
        df_train.to_parquet(train_path, index=False)
        df_test.to_parquet(test_path, index=False)

        if mlflow.active_run():
            mlflow.log_artifact(str(train_path))
            mlflow.log_artifact(str(test_path))
            print("✅ Feature-Splits gespeichert und geloggt.")

if __name__ == "__main__":
    main()
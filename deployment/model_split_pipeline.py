import os
import pandas as pd
import pyarrow.parquet as pq
import mlflow

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

TRAIN_YEARS = [2013, 2014]
TEST_YEARS = [2015, 2016]

FEATURES = ["hour", "weekday", "month", "year"]
TARGET = "trip_count"

def load_and_combine_data(years):
    dfs = []
    for year in years:
        for month in range(1, 13):
            filename = f"features_yellow_tripdata_{year}-{month:02}.parquet"
            path = os.path.join(DATA_DIR, filename)
            if os.path.exists(path):
                df = pd.read_parquet(path)
                df[FEATURES] = df[FEATURES].astype("int")
                dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=FEATURES + [TARGET])

def run_split_pipeline():
    mlflow.set_experiment("model_split_pipeline")
    with mlflow.start_run():
        train_df = load_and_combine_data(TRAIN_YEARS)
        test_df = load_and_combine_data(TEST_YEARS)

        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))

        train_path = os.path.join(DATA_DIR, "dataset_train.parquet")
        test_path = os.path.join(DATA_DIR, "dataset_test.parquet")

        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)

        mlflow.log_artifact(train_path)
        mlflow.log_artifact(test_path)
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("target", TARGET)

if __name__ == "__main__":
    run_split_pipeline()
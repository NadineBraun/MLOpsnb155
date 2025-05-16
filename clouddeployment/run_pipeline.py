import argparse
import mlflow
from data_pipeline import run_data_pipeline

mlflow.set_tracking_uri("file:/app/mlruns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=str, required=True)
    args = parser.parse_args()

    years = [int(y) for y in args.years.split(",")]
    run_data_pipeline(years)
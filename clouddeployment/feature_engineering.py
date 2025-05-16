import pandas as pd
from pathlib import Path
from tqdm import tqdm
import mlflow
import json
import hashlib

def get_column_hash(columns):
    col_string = ",".join(sorted(columns))
    return hashlib.md5(col_string.encode()).hexdigest()

def run_feature_engineering_for_years(years, cleaned_path="cleaned", output_path="features", data_path="data"):
    with open("mapping_overrides.json") as f:
        all_mappings = json.load(f)

    with mlflow.start_run(run_name="feature_engineering_all_years", nested=True):
        for year in tqdm(years, desc="Feature Engineering"):
            for month in range(1, 13):
                file = f"yellow_tripdata_{year}-{month:02d}.parquet"
                file_path = Path(cleaned_path) / f"cleaned_{file}"
                out_path = Path(output_path) / f"features_{file}"

                if not file_path.exists():
                    continue

                original_file = Path(data_path) / file
                if not original_file.exists():
                    print(f"❌ Originaldatei {original_file} nicht gefunden für Hash-Berechnung. Überspringe.")
                    continue
                original_df = pd.read_parquet(original_file)
                col_hash = get_column_hash(original_df.columns)
                df = pd.read_parquet(file_path)

                if col_hash not in all_mappings:
                    print(f"❌ Kein Mapping für Datei {file} mit Struktur-Hash {col_hash} gefunden. Überspringe.")
                    continue

                cols = all_mappings[col_hash]

                df["pickup_dt"] = pd.to_datetime(df[cols["pickup_col"]], errors="coerce")
                df["dropoff_dt"] = pd.to_datetime(df[cols["dropoff_col"]], errors="coerce")
                df["trip_duration"] = (df["dropoff_dt"] - df["pickup_dt"]).dt.total_seconds() / 60
                df["pickup_hour"] = df["pickup_dt"].dt.floor("H")

                df["payment_type"] = df[cols["payment_type_col"]].astype(str).str.lower() if "payment_type_col" in cols else ""

                agg = df.groupby("pickup_hour").agg(
                    trip_count=(cols["distance_col"], "count"),
                    total_distance=(cols["distance_col"], "sum"),
                    total_fare=(cols["fare_col"], "sum"),
                    total_amount=(cols["total_col"], "sum"),
                    total_passengers=(cols["passenger_col"], "sum"),
                    total_duration=("trip_duration", "sum"),
                    pct_credit_card=("payment_type", lambda x: (x == "credit").mean())
                ).reset_index()

                agg["hour"] = agg["pickup_hour"].dt.hour
                agg["weekday"] = agg["pickup_hour"].dt.weekday
                agg["month"] = agg["pickup_hour"].dt.month
                agg["year"] = agg["pickup_hour"].dt.year

                out_path.parent.mkdir(exist_ok=True)
                agg.to_parquet(out_path, index=False)
                mlflow.log_artifact(str(out_path))

        print("Feature Engineering abgeschlossen.")
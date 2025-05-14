import pandas as pd
from pathlib import Path
from tqdm import tqdm
import hashlib
import mlflow

column_cache = {}
structure_hashes = {}

def get_column_hash(columns):
    col_string = ",".join(sorted(columns))
    return hashlib.md5(col_string.encode()).hexdigest()

def detect_columns_with_prompt(columns, file, df_sample=None):
    col_hash = get_column_hash(columns)

    if col_hash in column_cache:
        col_map = column_cache[col_hash].copy()
    else:
        col_map = {
            "pickup_col": None,
            "dropoff_col": None,
            "distance_col": None,
            "fare_col": None,
            "total_col": None,
            "passenger_col": None,
            "payment_type_col": None,
            "pickup_long_col": None,
            "pickup_lat_col": None,
            "dropoff_long_col": None,
            "dropoff_lat_col": None
        }

        column_list = list(columns)
        sample_row = df_sample.iloc[0] if df_sample is not None and not df_sample.empty else {}

        for col in column_list:
            col_l = col.lower()
            if ("pickup" in col_l and "datetime" in col_l) or ("start" in col_l and "time" in col_l):
                col_map["pickup_col"] = col_map["pickup_col"] or col
            elif ("dropoff" in col_l and "datetime" in col_l) or ("end" in col_l and "time" in col_l):
                col_map["dropoff_col"] = col_map["dropoff_col"] or col
            elif "distance" in col_l:
                col_map["distance_col"] = col_map["distance_col"] or col
            elif "fare" in col_l:
                col_map["fare_col"] = col_map["fare_col"] or col
            elif "total" in col_l:
                col_map["total_col"] = col_map["total_col"] or col
            elif "passenger" in col_l:
                col_map["passenger_col"] = col_map["passenger_col"] or col
            elif "payment_type" in col_l:
                col_map["payment_type_col"] = col_map["payment_type_col"] or col
            elif "pickup_long" in col_l or "start_lon" in col_l:
                col_map["pickup_long_col"] = col_map["pickup_long_col"] or col
            elif "pickup_lat" in col_l or "start_lat" in col_l:
                col_map["pickup_lat_col"] = col_map["pickup_lat_col"] or col
            elif "dropoff_long" in col_l or "end_lon" in col_l:
                col_map["dropoff_long_col"] = col_map["dropoff_long_col"] or col
            elif "dropoff_lat" in col_l or "end_lat" in col_l:
                col_map["dropoff_lat_col"] = col_map["dropoff_lat_col"] or col

        for key, val in col_map.items():
            if val is None:
                print(f"\nHinweis: In Datei {file} konnte keine Spalte für '{key}' automatisch erkannt werden.")
                print("Verfügbare Spalten mit Beispielwerten:")
                for i, name in enumerate(column_list):
                    sample_val = sample_row.get(name, "-")
                    print(f"  [{i}] {name} → Beispielwert: {sample_val}")
                try:
                    index = int(input(f"Bitte Index der passenden Spalte für '{key}' eingeben (oder Enter zum Überspringen): "))
                    if 0 <= index < len(column_list):
                        col_map[key] = column_list[index]
                except ValueError:
                    print("Keine gültige Eingabe. Spalte wird übersprungen.")

        column_cache[col_hash] = col_map.copy()

    return col_map

def run_mapping_for_years(years, folder="."):
    results = []
    with mlflow.start_run(run_name="Column Mapping With Hash Support", nested=True):
        for year in tqdm(years, desc="Mapping Jahre"):
            for month in range(1, 13):
                file = f"yellow_tripdata_{year}-{month:02d}.parquet"
                file_path = Path(folder) / file
                if not file_path.exists():
                    continue
                try:
                    df = pd.read_parquet(file_path)
                    col_map = detect_columns_with_prompt(df.columns, file, df_sample=df)
                    col_map["file"] = file
                    results.append(col_map)
                except Exception as e:
                    print(f"Fehler bei {file}: {e}")

        mapping_df = pd.DataFrame(results)
        output_path = Path("cleaned") / "column_mapping_overview.csv"
        output_path.parent.mkdir(exist_ok=True)
        mapping_df.to_csv(output_path, index=False)
        mlflow.log_artifact(str(output_path))
        print("Spalten-Mapping abgeschlossen.")
        return mapping_df
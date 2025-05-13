import pandas as pd
from pathlib import Path
from tqdm import tqdm
import mlflow

MAX_SPEED = 100.0
REMOVE_ZERO_COORDS = True

def run_preprocessing_for_years(years, folder="."):
    mapping_df = pd.read_csv("cleaned/column_mapping_overview.csv")
    mapping_df["file"] = mapping_df["file"].str.strip()
    mapping_df = mapping_df.drop_duplicates(subset="file", keep="last")
    
    for year in tqdm(years, desc="Preprocessing Jahre"):
        for month in range(1, 13):
            file = f"yellow_tripdata_{year}-{month:02d}.parquet"
            file_path = Path(folder) / file
            if not file_path.exists():
                continue

            if file not in mapping_df["file"].values:
                print(f"Dateiname {file} nicht im Mapping enthalten. Verfügbare Einträge:")
                print(mapping_df["file"].unique())
                print(f"Kein Mapping für Datei {file} gefunden. Überspringe.")
                continue

            col_map_row = mapping_df[mapping_df["file"] == file]
            col_map = col_map_row.iloc[0].to_dict()

            with mlflow.start_run(run_name=f"preprocessing_{file}", nested=True):
                mlflow.log_param("file", file)
                mlflow.log_param("max_speed", MAX_SPEED)
                mlflow.log_param("remove_zero_coords", REMOVE_ZERO_COORDS)

                df = pd.read_parquet(file_path)
                original_count = len(df)

                df['trip_duration_min'] = (
                    pd.to_datetime(df[col_map['dropoff_col']]) - pd.to_datetime(df[col_map['pickup_col']])
                ).dt.total_seconds() / 60
                df['speed_mph'] = df[col_map['distance_col']] / (df['trip_duration_min'] / 60)
                
                filtered_total_amount = df[df[col_map["total_col"]] <= 0]

                filtered_zero = df[
                    (df['trip_duration_min'] <= 0) & (df[col_map['distance_col']] <= 0)
                ]

                filtered_coords = df[
                    ((df[col_map['pickup_long_col']] == 0) | (df[col_map['pickup_lat_col']] == 0)) &
                    ((df[col_map['distance_col']] <= 0) | (df[col_map['fare_col']] <= 0))
                ] if REMOVE_ZERO_COORDS else pd.DataFrame(columns=df.columns)

                filtered_speed = df[
                    (df['trip_duration_min'] <= 0) |
                    (df[col_map['fare_col']] <= 0) |
                    (df['speed_mph'] > MAX_SPEED)
                ]
                
                filtered_negative_distance = df[df[col_map['distance_col']] < 0]

                removed_df = pd.concat([filtered_coords, filtered_speed, filtered_total_amount, filtered_zero, filtered_negative_distance]).drop_duplicates()
                df_clean = df.loc[~df.index.isin(removed_df.index)]

                cleaned_count = len(df_clean)
                removed_rows = original_count - cleaned_count

                removed_path = Path("cleaned") / f"removed_{file}"
                removed_path.parent.mkdir(exist_ok=True)
                removed_df.to_parquet(removed_path, index=False)

                mlflow.log_metric("original_rows", original_count)
                mlflow.log_metric("cleaned_rows", cleaned_count)
                mlflow.log_metric("rows_removed", removed_rows)

                output_path = Path("cleaned") / f"cleaned_{file}"
                output_path.parent.mkdir(exist_ok=True)
                df_clean.to_parquet(output_path, index=False)

                mlflow.log_artifact(str(output_path))
                mlflow.log_artifact(str(removed_path))

    print("Preprocessing für alle angegebenen Jahre abgeschlossen.")
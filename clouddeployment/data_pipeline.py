from mapping import run_mapping_for_years
from preprocessing import run_preprocessing_for_years
from feature_engineering import run_feature_engineering_for_years
import mlflow

def run_data_pipeline(years, raw_data_path=".", cleaned_path="cleaned", output_path="features"):
    print("Starte Mapping ...")
    run_mapping_for_years(years, folder=raw_data_path)

    print("Starte Preprocessing ...")
    run_preprocessing_for_years(years, folder=raw_data_path)

    print("Starte Feature Engineering ...")
    run_feature_engineering_for_years(years, cleaned_path=cleaned_path, output_path=output_path)

    print("Datenverarbeitung abgeschlossen.")


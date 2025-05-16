import streamlit as st
st.set_page_config(page_title="NYC Taxi MLOps UI", layout="wide")

import subprocess
import mlflow
import os
import pandas as pd
import json
import pyarrow.parquet as pq
from pathlib import Path

st.title("🚕 NYC Yellow Taxi – MLOps Dashboard")

# --- TAB SETUP ---
tabs = st.tabs(["Manuelles Mapping", "Daten-Pipeline", "Daten-Split", "Modell-Training", "MLflow Explorer", "Tagesvorhersage", "Monitoring", "Drift-Analyse"])

# BASEDIR (anpassen, wenn nötig)
BASE_DIR = os.getcwd()

# --- TAB 2: Datenpipeline ---
with tabs[1]:
    st.header("Preprocessing und Feature Engineering starten")
    years = st.multiselect("Wähle Jahre für die Daten-Pipeline für das Preprocessing und Feature Engineering:", options=list(range(2013, 2017)), default=[2013, 2014, 2015, 2016])
    if st.button("Starte Datenpipeline"):
        years_str = ",".join(map(str, years))
        cmd = [
            "mlflow", "run", BASE_DIR,
            "--entry-point", "main",
            "--experiment-name", "default",
            "--storage-dir", os.path.join(BASE_DIR, "mlruns"),
            "--env-manager=local",
            "-P", f"years={years_str}"
        ]
        with st.spinner("Führe Datenpipeline aus..."):
            result = subprocess.run(cmd, capture_output=True, text=True)
            st.text(result.stdout)
            if result.returncode != 0:
                st.error(result.stderr)
            else:
                st.success("Datenpipeline erfolgreich ausgeführt!")

# --- TAB 3: Daten-Split ---
with tabs[2]:
    st.header("Daten-Split starten")
    train_years = st.multiselect("Trainingsjahre", options=list(range(2013, 2017)), default=[2013,2014], key="train_split")
    test_years = st.multiselect("Testjahre", options=list(range(2013, 2017)), default=[2015, 2016], key="test_split")

    if st.button("Starte Daten-Split"):
        train_str = ",".join(map(str, train_years))
        test_str = ",".join(map(str, test_years))

        split_cmd = [
            "mlflow", "run", BASE_DIR,
            "--entry-point", "model_split",
            "--experiment-name", "default",
            "--storage-dir", os.path.join(BASE_DIR, "mlruns"),
            "--env-manager", "local",
            "-P", f"train_years={train_str}",
            "-P", f"test_years={test_str}"
        ]
        with st.spinner("Führe Daten-Split aus..."):
            result_split = subprocess.run(split_cmd, capture_output=True, text=True)
            st.text(result_split.stdout)
            if result_split.returncode != 0:
                st.error("Fehler beim Split:")
                st.text(result_split.stderr)
            else:
                st.success("Daten-Split erfolgreich abgeschlossen!")

# --- TAB 4: Modell-Training ---
with tabs[3]:
    st.header("Modell-Training starten")
    target = st.text_input("Zielvariable", value="trip_count")
    st.caption("Genutzte Features für das Modell:")
    st.code("['hour', 'weekday', 'month', 'year']", language="python")

    if st.button("Starte Modelltraining"):
        train_cmd = [
            "mlflow", "run", BASE_DIR,
            "--entry-point", "model_training",
            "--experiment-name", "default",
            "--storage-dir", os.path.join(BASE_DIR, "mlruns"),
            "--env-manager", "local",
            "-P", f"train_path=df_train.parquet",
            "-P", f"test_path=df_test.parquet",
            "-P", f"target_col={target}"
        ]
        with st.spinner("Führe Modelltraining aus..."):
            result_train = subprocess.run(train_cmd, capture_output=True, text=True)
            st.text(result_train.stdout)
            if result_train.returncode != 0:
                st.error("Fehler beim Training:")
                st.text(result_train.stderr)
            else:
                st.success("Modell erfolgreich trainiert!")

# --- TAB 5: MLflow Explorer ---
with tabs[4]:
    st.header("MLflow Explorer")
    tracking_uri = mlflow.get_tracking_uri()
    st.caption(f"Tracking URI: {tracking_uri}")

    experiments = mlflow.search_experiments()
    exp_names = [e.name for e in experiments]
    selected_exp = st.selectbox("Experiment auswählen", exp_names)

    runs_df = mlflow.search_runs(experiment_names=[selected_exp])
    if not runs_df.empty:
        selected_run_id = st.selectbox("Run auswählen", runs_df["run_id"])
        run = mlflow.get_run(selected_run_id)

        st.subheader("📊 Metriken")
        st.json(run.data.metrics)

        st.subheader("⚙️ Parameter")
        st.json(run.data.params)

        st.subheader("📁 Artefakte")
        art_paths = mlflow.artifacts.download_artifacts(run_id=selected_run_id)
        st.text(f"Artefaktpfad: {art_paths}")
    else:
        st.info("Keine Runs im ausgewählten Experiment gefunden.")

# --- TAB 1: Manuelles Mapping ---
with tabs[0]:
    st.header("Manuelles Spalten-Mapping")

    years = st.multiselect(
        "Wähle Jahre für das Beispiel-Mapping:",
        options=list(range(2013, 2016)),
        default=[2013]
    )

    sample_file = None
    data_folder = Path("data")
    for year in years:
        test_path = data_folder / f"yellow_tripdata_{year}-01.parquet"
        if test_path.exists():
            sample_file = test_path
            break

    if sample_file:
        df = pq.read_table(sample_file).to_pandas().head(1)
        columns = df.columns.tolist()

        st.write("Verfügbare Spalten mit Beispielwerten aus:", sample_file)
        for col in columns:
            st.markdown(f"- **{col}** → Beispielwert: {df[col].iloc[0]}")

        pickup_long = st.selectbox("Spalte für pickup_long_col", columns)
        pickup_lat = st.selectbox("Spalte für pickup_lat_col", columns)
        dropoff_long = st.selectbox("Spalte für dropoff_long_col", columns)
        dropoff_lat = st.selectbox("Spalte für dropoff_lat_col", columns)
        pickup = st.selectbox("Spalte für pickup_col (Zeit)", columns)
        dropoff = st.selectbox("Spalte für dropoff_col (Zeit)", columns)
        distance = st.selectbox("Spalte für distance_col", columns)
        fare = st.selectbox("Spalte für fare_col", columns)
        total = st.selectbox("Spalte für total_col", columns)
        passenger = st.selectbox("Spalte für passenger_col", columns)
        payment = st.selectbox("Spalte für payment_type_col", columns)

        if st.button("Speichere Mapping-Datei"):
            from hashlib import md5
            col_hash = md5(",".join(sorted(columns)).encode()).hexdigest()
            mapping_path = Path("mapping_overrides.json")

            all_mappings = {}
            if mapping_path.exists():
                with open(mapping_path) as f:
                    all_mappings = json.load(f)

            all_mappings[col_hash] = {
                "pickup_col": pickup,
                "dropoff_col": dropoff,
                "distance_col": distance,
                "fare_col": fare,
                "total_col": total,
                "passenger_col": passenger,
                "payment_type_col": payment,
                "pickup_long_col": pickup_long,
                "pickup_lat_col": pickup_lat,
                "dropoff_long_col": dropoff_long,
                "dropoff_lat_col": dropoff_lat
            }

            with open(mapping_path, "w") as f:
                json.dump(all_mappings, f, indent=2)

            st.success(f"Mapping für Hash {col_hash} gespeichert in {mapping_path}.")
    else:
        st.warning("Keine Beispieldatei im gewählten Zeitraum gefunden.")

        
# --- TAB 6: Tagesvorhersage ---
with tabs[5]:
    st.header("Tagesvorhersage")
    input_date = st.date_input("Wähle ein Datum zur Vorhersage")

    if st.button("Starte Vorhersage"):
        from xgboost import XGBRegressor
        import datetime

        model_path = os.path.join(BASE_DIR, "model_output", "xgboost_model.json")
        if not os.path.exists(model_path):
            st.error("Kein trainiertes Modell gefunden.")
        else:
            model = XGBRegressor()
            model.load_model(model_path)

            date = input_date
            df_pred = pd.DataFrame({
                "hour": list(range(24)),
                "weekday": [date.weekday()] * 24,
                "month": [date.month] * 24,
                "year": [date.year] * 24
            })

            preds = model.predict(df_pred)
            df_pred["prediction"] = preds
            st.line_chart(df_pred.set_index("hour")["prediction"])
            st.success("Vorhersage abgeschlossen.")
            
# --- TAB 7: Monitoring ---
with tabs[6]:
    st.header("Monitoring / Referenzvergleich")
    reference_path = os.path.join(BASE_DIR, "reference_metrics.json")
    latest_run = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)

    if latest_run.empty:
        st.info("Kein MLflow-Run gefunden.")
    else:
        run_id = latest_run.iloc[0]["run_id"]
        run = mlflow.get_run(run_id)

        st.subheader("📁 Aktuelle Metriken")
        st.json(run.data.metrics)

        if os.path.exists(reference_path):
            st.subheader("📁 Referenzmetriken")
            with open(reference_path) as f:
                ref = json.load(f)
            st.json(ref)

            if "R2" in run.data.metrics and "R2" in ref:
                delta = run.data.metrics["R2"] - ref["R2"]
                st.metric("R² Delta", f"{delta:.3f}", delta_color="inverse")
        else:
            st.info("Keine Referenzdatei gefunden. Erstelle eine unter 'reference_metrics.json'")
            
# --- TAB 8: Drift-Analyse ---
with tabs[7]:
    st.header("Monitoring: Data & Concept Drift")

    import os
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from xgboost import XGBRegressor
    from scipy.stats import ks_2samp

    def population_stability_index(expected, actual, buckets=10):
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        psi = np.sum((expected_counts - actual_counts) * np.log((expected_counts + 1e-6) / (actual_counts + 1e-6)))
        return psi

    def interpret_psi(psi):
        if psi < 0.1:
            return "Kein Drift erkannt (PSI < 0.1).", "info"
        elif psi < 0.25:
            return "Leichter Drift erkannt (0.1 ≤ PSI < 0.25).", "warning"
        else:
            return "Starker Drift erkannt (PSI ≥ 0.25).", "error"

    years = list(range(2013, 2017))
    years_1 = st.multiselect("Zeitraum 1: Wähle Jahre", years, default=[2013, 2014], key="zeitraum1")
    years_2 = st.multiselect("Zeitraum 2: Wähle Jahre", years, default=[2015, 2016], key="zeitraum2")
    selected_feature = st.selectbox("Temporales Merkmal", ["hour", "weekday", "month", "year"])

    def load_data(years):
        dfs = []
        for year in years:
            for month in range(1, 13):
                file = f"features_yellow_tripdata_{year}-{month:02d}.parquet"
                path = os.path.join("features", file)
                if os.path.exists(path):
                    df = pd.read_parquet(path)
                    df["year"] = year
                    dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if st.button("Starte Monitoring"):
        df1 = load_data(years_1)
        df2 = load_data(years_2)

        if df1.empty or df2.empty:
            st.warning("Zeitraum 1 oder 2 enthält keine gültigen Daten. Bitte wähle andere Jahre.")
        else:
            st.subheader("1️⃣ Data Drift: trip_count über Zeitmerkmale")

            g1 = df1.groupby(selected_feature)["trip_count"].mean()
            g2 = df2.groupby(selected_feature)["trip_count"].mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=g1.index, y=g1.values, mode='lines+markers', name='Zeitraum 1'))
            fig.add_trace(go.Scatter(x=g2.index, y=g2.values, mode='lines+markers', name='Zeitraum 2'))
            fig.update_layout(title=f"Mittlere trip_count über {selected_feature}", xaxis_title=selected_feature, yaxis_title="trip_count")
            st.plotly_chart(fig, use_container_width=True)

            combined = pd.DataFrame({
                "trip_count_1": df1[selected_feature].map(df1.groupby(selected_feature)["trip_count"].mean()),
                "trip_count_2": df2[selected_feature].map(df2.groupby(selected_feature)["trip_count"].mean())
            }).dropna()

            ks_stat, p_val = ks_2samp(combined["trip_count_1"], combined["trip_count_2"])
            psi_val = population_stability_index(combined["trip_count_1"], combined["trip_count_2"])

            st.markdown(f"**KS-Test**: D = {ks_stat:.4f}, p-Wert = {p_val:.4f}")
            st.markdown(f"**PSI (Population Stability Index)**: {psi_val:.4f}")

            psi_message, msg_type = interpret_psi(psi_val)
            if msg_type == "info":
                st.info(psi_message)
            elif msg_type == "warning":
                st.warning(psi_message)
            elif msg_type == "error":
                st.error(psi_message)

            st.subheader("2️⃣ Concept Drift: Analyse der Modellresiduen")

            model_path = os.path.join(BASE_DIR, "model_output", "xgboost_model.json")
            if not os.path.exists(model_path):
                st.warning("Kein Modell gefunden. Bitte trainiere und speichere zuerst ein Modell.")
            else:
                model = XGBRegressor()
                model.load_model(model_path)

                features = ["hour", "weekday", "month", "year"]
                for df in [df1, df2]:
                    df["prediction"] = model.predict(df[features])
                    df["residual"] = df["trip_count"] - df["prediction"]

                residuals_1 = df1["residual"].dropna()
                residuals_2 = df2["residual"].dropna()

                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(x=residuals_1, opacity=0.6, name="Zeitraum 1"))
                fig2.add_trace(go.Histogram(x=residuals_2, opacity=0.6, name="Zeitraum 2"))
                fig2.update_layout(barmode="overlay", title="Histogramm der Residuen", xaxis_title="Residual", yaxis_title="Anzahl")
                st.plotly_chart(fig2, use_container_width=True)

                ks_r, p_r = ks_2samp(residuals_1, residuals_2)
                psi_r = population_stability_index(residuals_1, residuals_2)

                st.markdown(f"**KS-Test auf Residuen**: D = {ks_r:.4f}, p-Wert = {p_r:.4f}")
                st.markdown(f"**PSI auf Residuen**: {psi_r:.4f}")

                psi_message_r, msg_type_r = interpret_psi(psi_r)
                if msg_type_r == "info":
                    st.info(psi_message_r)
                elif msg_type_r == "warning":
                    st.warning(psi_message_r)
                elif msg_type_r == "error":
                    st.error(psi_message_r)
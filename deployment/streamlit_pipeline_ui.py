import streamlit as st
import subprocess
import sys
import json
import pathlib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="NYC Taxi MLOps UI", layout="wide")
st.title("🚕 NYC Yellow Taxi – MLOps Dashboard")

# Verzeichnis und Skriptpfade definieren
deployment_dir = pathlib.Path(__file__).parent
split_script = deployment_dir / "model_split_pipeline.py"
train_script = deployment_dir / "model_pipeline.py"
metrics_file = deployment_dir / "model_output" / "metrics.json"

# Tabs
tabs = st.tabs([
    "Daten-Split",
    "Modell-Training",
    "Tagesvorhersage",
    "Monitoring",
    "Drift-Analyse"
])

# --- Tab 1: Daten-Split ---
with tabs[0]:
    st.header("Daten-Split durchführen (2013/14 → train, 2015/16 → test)")

    if st.button("Starte Daten-Split"):
        with st.spinner("Führe Split-Pipeline aus..."):
            result = subprocess.run([sys.executable, str(split_script)], capture_output=True, text=True)
            st.text(result.stdout)
            if result.returncode != 0:
                st.error(result.stderr)
            else:
                st.success("Daten-Split erfolgreich abgeschlossen.")

# --- Tab 2: Modell-Training ---
with tabs[1]:
    st.header("Modell trainieren und mit MLflow loggen")

    if st.button("Starte Training"):
        with st.spinner("Trainiere Modell..."):
            result = subprocess.run([sys.executable, str(train_script)], capture_output=True, text=True)
            st.text(result.stdout)
            if result.returncode != 0:
                st.error(result.stderr)
            else:
                st.success("Modelltraining erfolgreich abgeschlossen.")

# --- Tab 3: Tagesvorhersage (Platzhalter) ---
with tabs[2]:
    st.header("Tagesvorhersage")
    st.info("Diese Funktion ist aktuell nicht implementiert.")

# --- Tab 4: Monitoring (Metriken anzeigen) ---
with tabs[3]:
    st.header("Monitoring – letzte Modellmetriken")

    try:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{metrics['rmse']:.3f}")
        col2.metric("MAE", f"{metrics['mae']:.3f}")
        col3.metric("R²", f"{metrics['r2']:.3f}")

        st.success("Metriken erfolgreich geladen.")
    except FileNotFoundError:
        st.warning("Noch keine Metriken gefunden. Bitte zuerst das Modell trainieren.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Metriken: {e}")

# --- Tab 5: Drift-Analyse (Platzhalter) ---
with tabs[4]:
    st.header("Drift-Analyse")
    st.info("Diese Funktion ist aktuell nicht implementiert.")

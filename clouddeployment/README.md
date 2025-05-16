# NYC Yellow Taxi – MLOps Deployment

Dieser Ordner enthält alle relevanten Skripte und Konfigurationen zur Ausführung einer MLOps-Pipeline mit Streamlit und MLflow für die NYC Yellow Taxi Daten. Die Lösung ist containerisiert über Docker und lauffähig auf einem beliebigen Linux-Server (z. B. Hetzner Cloud).

## Projektstruktur

```plaintext
clouddeployment/
├── Dockerfile                # Docker-Konfiguration
├── requirements.txt          # Alle Python-Abhängigkeiten
├── .dockerignore             # Ausgeschlossene Dateien für Docker
├── MLproject                 # MLflow-Projektdefinition
├── mapping_overrides.json    # Manuell gepflegte Spaltenzuordnung
├── run_pipeline.py           # Hauptsteuerung der gesamten Pipeline
├── data_pipeline.py          # Ansteuerung der Mapping-, Preprocessing- und Feature-Pipeline
├── preprocessing.py          # Datenbereinigung mit Spalten-Mapping
├── feature_engineering.py    # Aggregation von Zeitreihenmerkmalen
├── split_pipeline.py         # Aufteilung in Trainings- und Testdaten
├── model_pipeline.py         # Modelltraining mit XGBoost + MLflow
├── streamlit_pipeline_ui.py  # Benutzeroberfläche mit Streamlit
├── reference_metrics.json    # (Optional) Referenzwerte für Monitoring
└── data/                     # (Nicht enthalten) Ordner für die .parquet-Dateien
└── cleaned/                  # (Nicht enthalten) Ordner für die .parquet-Dateien nach dem Preprocessing
```

## Hinweis zum Ordner `data/` und `cleaned/`

Der Ordner `data/`, der die `.parquet`-Dateien für die Jahre 2013–2016 enthält sowie der Ordner /cleaned/´, der die Dateien cleaned und removed für die entsprechenden Jahre enthält, wurden **nicht in dieses Repository aufgenommen**, da diese sehr groß sind und sich daher nicht für GitHub oder einfache Übertragungen eignen.

Die Skripte erwarten Dateien mit folgendem Namensschema:
>
> ```
> yellow_tripdata_YYYY-MM.parquet (Rohdaten)
> ```
> ```
> cleaned_yellow_tripdata_YYYY-MM.parquet (Dateien zur Weiterverarbeitung im weiteren Flow nach dem Preprocessing)
> ```
> ´´´
> removed_yellow_tripdata_YYYY-MM.parquet (Dateien mit den Zeilen, die beim Preprocessing entfernt wurden)
> ```


## Live-Zugriff

Das Deployment wird über einen kostenpflichtigen Hetzner Cloud Server mit bereitgestellt und ist über die nachfolgende Adresse für einen Monat erreichbar:

```
http://91.99.8.55:8501
```

Du kannst die Anwendung ohne Installation oder Zugangsdaten direkt im Browser aufrufen.

> **Hinweis:** Aus Kostengründen wurde die deployte Version auf die Jahre 2013 bis 2016 beschränkt.  
> Eine vollständige Verarbeitung aller verfügbaren NYC Yellow Taxi Daten wäre technisch möglich, wurde aber aus Effizienzgründen nicht umgesetzt.

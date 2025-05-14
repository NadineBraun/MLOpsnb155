# MLOpsnb155

# NYC Yellow Taxi – MLOps-Projekt

Dieses Repository dokumentiert die Entwicklung und Umsetzung eines prototypischen Machine-Learning-Workflows zur Vorhersage von Taxinachfrage auf Basis der Yellow Taxi Trip Records der Stadt New York.

Ziel war es, zentrale Bausteine eines MLOps-Prozesses – von der Datenvorverarbeitung über das Modelltraining bis hin zum Monitoring – exemplarisch zu implementieren und als interaktive Anwendung bereitzustellen.

---

## Struktur des Repositories

### Zentrale Dokumentation

- `Machine Learning Operations Projekt NY Yellow Taxi.ipynb` – Jupyter Notebook mit dem gesamten Entwicklungsprozess
- `Machine Learning Operations Projekt NY Yellow Taxi.html` – gerenderte HTML-Version des Notebooks
- Enthält neben Code auch begleitende Erläuterungen, Visualisierungen und die textliche Ausformulierung des ML Canvas

---

### Pipelines

- `pipelines_terminal/`  
  Enthält die modularen Skripte für die lokale Ausführung über das Terminal. Dazu zählen Datenmapping, Preprocessing, Feature Engineering, Modelltraining und Logging mit MLflow.

- `pipelines_streamlit_local/`  
  Beinhaltet die Streamlit-Variante zur lokalen Nutzung mit grafischer Benutzeroberfläche. Hier werden Teile der Pipeline interaktiv steuerbar gemacht.

---

###  Cloud-Deployment

- `deployment/`  
  Enthält die Dateien für die cloudbasierte Variante der Anwendung, wie sie auf **Streamlit Cloud** veröffentlicht wurde. Diese Version bildet exemplarisch Modelltraining, Metriklogging und Monitoring ab und kann direkt im Browser genutzt werden – ohne lokale Installation.

---

### Beispieldateien

- `Beispieldaten/`  
  Beispielhafte `.parquet`-Dateien, die entweder aus dem Jupyter Notebook erzeugt oder zur Demonstration verwendet wurden.  
  Da der vollständige Datensatz weit über 100 GB umfasst, sind einige Dateien hier in reduzierter Form enthalten (z. B. auf 1000 Zeilen gekürzt), um die GitHub-Größenbeschränkung einzuhalten.

---

### Begleitmaterial

- `data_dictionary_trip_records_yellow.pdf` – offizielles Datenverzeichnis der Yellow Taxi Trip Records
- `ML_Ops_NYYellowTaxi_nb155.pptx` – Projektpräsentation mit Überblick über Zielsetzung, Vorgehen und zentrale Ergebnisse

---

## Hinweise

- Bei Bedarf können sämtliche Originaldaten sowie weitere Zwischenergebnisse zur Verfügung gestellt werden. Aufgrund ihrer Größe konnten sie jedoch nicht im Repository abgelegt werden.
- Die Anwendung nutzt ausschließlich Open Data der Stadt New York. Die Daten wurden vorab aggregiert und auf Stundenebene zusammengeführt.

---

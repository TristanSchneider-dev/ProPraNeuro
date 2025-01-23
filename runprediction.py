import os
import pandas as pd
import pickle
import json
import sys

#python testModel.py 2023-05-01 2023-05-31 /home/pete-linux/Downloads/1_train_and_test.csv mm

start_date = sys.argv[1]  # Startdatum
end_date = sys.argv[2]    # Enddatum
csv_path = sys.argv[3]    # Pfad zur CSV-Datei
unit = sys.argv[4]        # Einheit der Werte

config = {
    "start_date": start_date,  # Startdatum für den Zeitraum
    "end_date": end_date,  # Enddatum für den Zeitraum
    "csv_path": csv_path,  # Pfad zur CSV-Datei
    "csv_columns": ["index", "date", "value", "unit"],  # Spaltennamen in der CSV
    "csv_sep": ";",  # Trennzeichen in der CSV
    "drop_columns": ["index"],  # Spalten, die entfernt werden sollen
    "datetime_unit": "ms",  # Einheit der Zeitstempel (Millisekunden)
    "tail_rows": 1_000_000,  # Anzahl der letzten Zeilen, die geladen werden
    "unit": unit  # Einheit der Werte
}

model_save_path = os.path.splitext(config["csv_path"])[0] + ".pkl"

def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df.loc[:, 'hour'] = df['date'].dt.hour
    df.loc[:, 'dayofweek'] = df['date'].dt.dayofweek
    df.loc[:, 'quarter'] = df['date'].dt.quarter
    df.loc[:, 'month'] = df['date'].dt.month
    df.loc[:, 'year'] = df['date'].dt.year
    df.loc[:, 'dayofyear'] = df['date'].dt.dayofyear
    df.loc[:, 'dayofmonth'] = df['date'].dt.day

    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth']]
    if label:
        y = df[label]
        return X, y
    return X

# Daten vorbereiten
date_range = pd.date_range(start=config["start_date"], end=config["end_date"], freq='h')
may_dates = pd.DataFrame({'date': date_range})

# Merkmale für den Mai erstellen
X_may = create_features(may_dates)

# CSV-Datei laden
allCSV = pd.read_csv(
    config["csv_path"],
    names=config["csv_columns"],
    sep=config["csv_sep"],
    header=1
)
allCSV["date"] = pd.to_datetime(allCSV["date"], unit=config["datetime_unit"])

# Nicht benötigte Spalten entfernen
for col in config["drop_columns"]:
    allCSV.pop(col)

# Datensatz beschränken
allCSV = allCSV.tail(config["tail_rows"])

# Features und Labels für das Training/Testen erstellen
X_test, y_test = create_features(allCSV, label='value')

# Modell laden
with open(model_save_path, "rb") as file:
    reg_loaded = pickle.load(file)

# Vorhersagen für Mai-Daten
newPred = reg_loaded.predict(X_may)
may_dates['predicted_value'] = newPred

# Konvertiere Vorhersagen in JSON-Format
result = []
for _, row in may_dates.iterrows():
    messwert = {
        "index": None,  # Kein Index für die Vorhersagewerte
        "time": row['date'].isoformat(),
        "value": row['predicted_value'],
        "unit": config["unit"],  # Einheit aus der Config
        "color": "blue"  # Standardfarbe Blau
    }
    result.append(messwert)

# JSON-Ausgabe erzeugen
json_output = json.dumps(result, indent=4)
print(json_output)

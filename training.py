import os
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle

from sklearn.model_selection import train_test_split
plt.style.use('fivethirtyeight')

# Globale Konfiguration
config = {
    "csv_path": "/home/pete-linux/Downloads/3_dataset.csv",
    "csv_columns": ["index", "date", "value", "unit"],
    "csv_sep": ";",
    "drop_columns": ["index", "unit"],
    "datetime_unit": "ms",
    "test_size": 0.2,  # Anteil der Daten für den Testdatensatz (20%)
    "random_state": 42,  # Zufallswert für die Reproduzierbarkeit des Daten-Splits
    "model_params": {  # Parameter für das XGBoost-Modell
        "n_estimators": 100,
        "tree_method": "hist",
        "device": "cpu"
    }
}

# Dynamischer Modellpfad
model_save_path = os.path.splitext(config["csv_path"])[0] + ".pkl"

# Daten laden
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


def create_features(df, label=None):
    """
    Erstellt zeitbasierte Merkmale aus einem Datumsindex.

    Parameter:
        df: DataFrame mit einer Datums-Spalte
        label: Name der Zielvariable (falls vorhanden)

    Rückgabe:
        X: DataFrame mit Merkmalen
        y (optional): Zielvariable (falls label angegeben)
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


# Merkmale und Zielvariable aus den Daten extrahieren
X, y = create_features(allCSV, label='value')

# Aufteilen der Daten in Trainings- und Testdatensätze
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["test_size"],  # Anteil der Daten, der für Tests genutzt wird
    random_state=config["random_state"]  # Zufallswert für reproduzierbare Ergebnisse
)

# Initialisieren des XGBoost-Modells mit den angegebenen Parametern
reg = xgb.XGBRegressor(**config["model_params"])

# Modellparameter ausgeben (zur Kontrolle)
print("Modellparameter:", reg.get_params())

# Modell mit Trainingsdaten trainieren
reg.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],  # Überwachen der Leistung auf Train/Test
    verbose=True  # Ausgabe während des Trainings
)

# Modell speichern
with open(model_save_path, "wb") as file:
    pickle.dump(reg, file)

# Ausgabe des Speicherorts des Modells
print(f"Modell wurde unter '{model_save_path}' gespeichert.")

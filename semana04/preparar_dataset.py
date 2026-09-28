import pandas as pd
def calcular_dia_lluvioso(precipitacion):
    """
    Devuelve 1 si hubo precipitación,
    0 en caso contrario.
    """
    return int(precipitacion > 0)

# 1. Cargar el dataset
df = pd.read_csv("pronostico_huancayo.csv")

print("Dimensiones del dataset:", df.shape)

print("\nColumnas disponibles:")
print(df.columns.tolist())

print("\nPrimeras filas:")
print(df.head())

print("\nValores nulos por columna:")
print(df.isnull().sum())
# 2. Crear la variable objetivo
df["dia_lluvioso"] = (
    df["precipitacion"]
    .apply(calcular_dia_lluvioso)
)
df["amplitud_termica"] = df["temp_max"] - df["temp_min"]
columnas_features = [
    "temp_max",
    "temp_min",
    "amplitud_termica"
]

X = df[columnas_features].copy()

y = df["dia_lluvioso"].copy()

print("\nDistribución de la variable objetivo:")
print(y.value_counts())

print("\nEstadísticas:")
print(X.describe())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
antes = len(X)

datos_completos = pd.concat([X, y], axis=1).dropna()

X = datos_completos[columnas_features]
y = datos_completos["dia_lluvioso"]

print(
    f"\nFilas eliminadas por valores nulos: "
    f"{antes - len(X)}"
)
X_train, X_test, y_train, y_test = train_test_split(
    
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
escalador = StandardScaler()

X_train_esc = escalador.fit_transform(X_train)

X_test_esc = escalador.transform(X_test)
print(f"\nTamaño de entrenamiento: {X_train_esc.shape}")
print(f"Tamaño de prueba: {X_test_esc.shape}")

np.savez(
    "dataset_preparado.npz",
    X_train=X_train_esc,
    X_test=X_test_esc,
    y_train=y_train.values,
    y_test=y_test.values
)

print("\nArchivo dataset_preparado.npz guardado correctamente.")
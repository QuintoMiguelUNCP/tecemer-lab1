from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = make_classification(
    n_samples=500,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

escalador = StandardScaler()

X_train = escalador.fit_transform(X_train)
X_test = escalador.transform(X_test)

print(
    "X_train:", X_train.shape,
    "X_test:", X_test.shape
)
from tensorflow import keras
from tensorflow.keras import layers
modelo = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

modelo.summary()
modelo.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
historial = modelo.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
fig, ejes = plt.subplots(
    1,
    2,
    figsize=(10, 4)
)

ejes[0].plot(
    historial.history["loss"],
    label="entrenamiento"
)

ejes[0].plot(
    historial.history["val_loss"],
    label="validación"
)

ejes[0].set_title("Pérdida (loss)")
ejes[0].set_xlabel("Época")
ejes[0].legend()

ejes[1].plot(
    historial.history["accuracy"],
    label="entrenamiento"
)

ejes[1].plot(
    historial.history["val_accuracy"],
    label="validación"
)

ejes[1].set_title("Precisión (accuracy)")
ejes[1].set_xlabel("Época")
ejes[1].legend()

plt.tight_layout()

plt.savefig(
    "curvas_entrenamiento_sintetico.png",
    dpi=100
)

print(
    "\nGráfico guardado en "
    "curvas_entrenamiento_sintetico.png"
)
perdida_test, precision_test = modelo.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"\nPérdida en test: {perdida_test:.4f}"
)

print(
    f"Precisión en test: {precision_test:.4f}"
)
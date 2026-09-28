import tensorflow as tf
import numpy as np

# Dataset sencillo:
# y = 2*x1 - 3*x2 + ruido
rng = np.random.default_rng(42)
X = rng.normal(size=(500, 2)).astype("float32")
y = (2 * X[:, 0] - 3 * X[:, 1] + 0.2 * rng.normal(size=500)).astype("float32")

def crear_modelo(activacion):
    return tf.keras.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(16, activation=activacion),
        tf.keras.layers.Dense(8, activation=activacion),
        tf.keras.layers.Dense(1, activation="linear")
    ])

activaciones = {
    "relu": "relu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "gelu": tf.keras.activations.gelu,
    "swish": tf.keras.activations.swish,
    "mish": tf.keras.activations.mish,
}

for nombre, activacion in activaciones.items():
    tf.keras.backend.clear_session()

    modelo = crear_modelo(activacion)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="mse"
    )

    historial = modelo.fit(
        X, y,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    print(
        f"{nombre:8s} -> pérdida final: "
        f"{historial.history['loss'][-1]:.4f}"
    )

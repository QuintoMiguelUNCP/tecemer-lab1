import tensorflow as tf

entrada = tf.constant([[-4.0, -1.0, 0.0, 1.0, 4.0]])

# Uso directo de la activación.
salida = tf.keras.activations.sigmoid(entrada)

print("Entrada:")
print(entrada.numpy())
print("Sigmoide:")
print(salida.numpy())

# Ejemplo dentro de una red para clasificación binaria.
modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

dato = tf.constant([[0.2, 0.8, -0.4, 1.0]])
probabilidad = modelo(dato, training=False)

print("Probabilidad producida por el modelo:")
print(probabilidad.numpy())

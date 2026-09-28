import tensorflow as tf

entrada = tf.constant([[-3.0, -1.0, 0.0, 1.0, 3.0]])
salida = tf.keras.activations.linear(entrada)

print("Entrada:")
print(entrada.numpy())
print("Identidad / Linear:")
print(salida.numpy())

# En regresión es habitual usar una salida lineal.
modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear")
])

dato = tf.constant([[2.0, 3.0, -1.0]])
prediccion = modelo(dato, training=False)

print("Predicción de regresión:")
print(prediccion.numpy())

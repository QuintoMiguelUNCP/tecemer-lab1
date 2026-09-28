import tensorflow as tf

entrada = tf.constant([[-3.0, -1.0, 0.0, 1.0, 3.0]])
salida = tf.keras.activations.tanh(entrada)

print("Entrada:")
print(entrada.numpy())
print("Tanh:")
print(salida.numpy())

modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(5, activation="tanh"),
    tf.keras.layers.Dense(1)
])

dato = tf.constant([[0.5, -0.3, 0.8]])
print("Salida del modelo:")
print(modelo(dato, training=False).numpy())

import tensorflow as tf

entrada = tf.constant([[-3.0, -1.0, 0.0, 1.0, 3.0]])
salida = tf.keras.activations.mish(entrada)

print("Entrada:")
print(entrada.numpy())
print("Mish:")
print(salida.numpy())

modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(8, activation=tf.keras.activations.mish),
    tf.keras.layers.Dense(1)
])

dato = tf.constant([[0.4, -0.8, 1.2, 0.1]])
print("Salida del modelo:")
print(modelo(dato, training=False).numpy())

import tensorflow as tf

entrada = tf.constant([[-3.0, -1.0, 0.0, 1.0, 3.0]])
salida = tf.keras.activations.swish(entrada)

print("Entrada:")
print(entrada.numpy())
print("Swish / SiLU:")
print(salida.numpy())

modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(8, activation=tf.keras.activations.swish),
    tf.keras.layers.Dense(1)
])

dato = tf.constant([[1.0, 0.5, -1.0, 2.0]])
print("Salida del modelo:")
print(modelo(dato, training=False).numpy())

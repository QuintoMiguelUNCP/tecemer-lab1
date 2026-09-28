import tensorflow as tf

entrada = tf.constant([[-4.0, -2.0, 0.0, 2.0, 4.0]])

# En Keras LeakyReLU puede utilizarse como una capa.
leaky_relu = tf.keras.layers.LeakyReLU(negative_slope=0.1)
salida = leaky_relu(entrada)

print("Entrada:")
print(entrada.numpy())
print("Leaky ReLU:")
print(salida.numpy())

modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(8),
    tf.keras.layers.LeakyReLU(negative_slope=0.1),
    tf.keras.layers.Dense(1)
])

dato = tf.constant([[1.0, -1.0, 2.0, -2.0]])
print("Salida del modelo:")
print(modelo(dato, training=False).numpy())

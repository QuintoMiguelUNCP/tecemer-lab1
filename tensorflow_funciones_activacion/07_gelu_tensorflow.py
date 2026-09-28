import tensorflow as tf

entrada = tf.constant([[-3.0, -1.0, 0.0, 1.0, 3.0]])
salida = tf.keras.activations.gelu(entrada)

print("Entrada:")
print(entrada.numpy())
print("GELU:")
print(salida.numpy())

# GELU es muy utilizada en arquitecturas tipo Transformer.
modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(16, activation=tf.keras.activations.gelu),
    tf.keras.layers.Dense(1)
])

dato = tf.constant([[0.2, -0.7, 1.1, 0.4]])
print("Salida del modelo:")
print(modelo(dato, training=False).numpy())

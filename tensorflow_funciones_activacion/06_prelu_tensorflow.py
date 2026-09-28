import tensorflow as tf

# PReLU tiene un parámetro alpha entrenable.
prelu = tf.keras.layers.PReLU(
    alpha_initializer=tf.keras.initializers.Constant(0.25)
)

entrada = tf.constant([[-4.0, -2.0, 0.0, 2.0, 4.0]])
salida = prelu(entrada)

print("Entrada:")
print(entrada.numpy())
print("PReLU:")
print(salida.numpy())
print("Alpha inicial:")
print(prelu.get_weights()[0])

# Ejemplo dentro de un modelo.
modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(8),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Dense(1)
])

dato = tf.constant([[0.5, -1.0, 1.5, -0.2]])
print("Salida del modelo:")
print(modelo(dato, training=False).numpy())

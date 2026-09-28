import tensorflow as tf

# La función escalón no es diferenciable en x=0 y su gradiente es 0
# casi en todas partes, por eso se usa aquí solo como demostración.

def escalon(x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    return tf.where(x >= 0.0, 1.0, 0.0)

entrada = tf.constant([[-2.0, -0.5, 0.0, 1.0, 3.0]])
salida = escalon(entrada)

print("Entrada:")
print(entrada.numpy())
print("Salida con escalón:")
print(salida.numpy())

# También puede envolverse en una capa Lambda.
modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(5,)),
    tf.keras.layers.Lambda(escalon)
])

print("Salida del modelo:")
print(modelo(entrada).numpy())

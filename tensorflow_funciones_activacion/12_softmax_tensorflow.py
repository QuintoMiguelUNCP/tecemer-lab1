import tensorflow as tf

logits = tf.constant([[2.0, 1.0, 0.1]])
probabilidades = tf.keras.activations.softmax(logits)

print("Logits:")
print(logits.numpy())
print("Probabilidades Softmax:")
print(probabilidades.numpy())
print("Suma:")
print(tf.reduce_sum(probabilidades, axis=-1).numpy())

# Ejemplo para clasificación multiclase de 3 clases.
modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

dato = tf.constant([[0.2, 1.0, -0.5, 0.7]])
prediccion = modelo(dato, training=False)

print("Probabilidades del modelo:")
print(prediccion.numpy())
print("Clase predicha:")
print(tf.argmax(prediccion, axis=1).numpy())

import tensorflow as tf

class Maxout(tf.keras.layers.Layer):
    """
    Capa Maxout personalizada.

    Para cada unidad genera varias respuestas lineales y conserva
    el valor máximo entre ellas.
    """

    def __init__(self, units, pieces=2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.pieces = pieces
        self.proyeccion = tf.keras.layers.Dense(units * pieces)

    def call(self, inputs):
        z = self.proyeccion(inputs)

        # (batch, units * pieces) -> (batch, units, pieces)
        z = tf.reshape(z, (-1, self.units, self.pieces))

        # Selecciona el máximo de cada conjunto.
        return tf.reduce_max(z, axis=-1)

modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    Maxout(units=6, pieces=3),
    tf.keras.layers.Dense(1)
])

dato = tf.constant([[1.0, -2.0, 0.5, 3.0]])

salida = modelo(dato, training=False)

print("Entrada:")
print(dato.numpy())
print("Salida del modelo con Maxout:")
print(salida.numpy())

print("\nResumen del modelo:")
modelo.summary()

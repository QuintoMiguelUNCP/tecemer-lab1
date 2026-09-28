import tensorflow as tf

x = tf.constant([[-3.0, -1.0, 0.0, 1.0, 3.0]])

def escalon(x):
    return tf.where(x >= 0.0, 1.0, 0.0)

activaciones = {
    "Escalón": escalon(x),
    "Sigmoide": tf.keras.activations.sigmoid(x),
    "Tanh": tf.keras.activations.tanh(x),
    "ReLU": tf.keras.activations.relu(x),
    "GELU": tf.keras.activations.gelu(x),
    "Swish": tf.keras.activations.swish(x),
    "Mish": tf.keras.activations.mish(x),
    "Identidad": tf.keras.activations.linear(x),
}

leaky = tf.keras.layers.LeakyReLU(negative_slope=0.1)
activaciones["Leaky ReLU"] = leaky(x)

prelu = tf.keras.layers.PReLU(
    alpha_initializer=tf.keras.initializers.Constant(0.25)
)
activaciones["PReLU"] = prelu(x)

for nombre, salida in activaciones.items():
    print(f"\n{nombre}")
    print(salida.numpy())

logits = tf.constant([[2.0, 1.0, 0.1]])
print("\nSoftmax")
print(tf.keras.activations.softmax(logits).numpy())


class Maxout(tf.keras.layers.Layer):
    def __init__(self, units, pieces=2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.pieces = pieces
        self.dense = tf.keras.layers.Dense(units * pieces)

    def call(self, inputs):
        z = self.dense(inputs)
        z = tf.reshape(z, (-1, self.units, self.pieces))
        return tf.reduce_max(z, axis=-1)


maxout = Maxout(units=2, pieces=3)
dato = tf.constant([[1.0, -2.0, 0.5]])
print("\nMaxout")
print(maxout(dato).numpy())

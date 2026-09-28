EJEMPLOS TENSORFLOW / KERAS - FUNCIONES DE ACTIVACIÓN

Requisito:
    pip install tensorflow

Archivos:
01_escalon_tensorflow.py
02_sigmoide_tensorflow.py
03_tanh_tensorflow.py
04_relu_tensorflow.py
05_leaky_relu_tensorflow.py
06_prelu_tensorflow.py
07_gelu_tensorflow.py
08_swish_tensorflow.py
09_mish_tensorflow.py
10_maxout_tensorflow.py
11_identidad_tensorflow.py
12_softmax_tensorflow.py
13_todas_tensorflow.py
14_entrenamiento_comparativo.py

Ejemplo:
    python 04_relu_tensorflow.py

IMPORTANTE:
- Escalón se implementa manualmente porque no es adecuada para
  entrenamiento por retropropagación.
- PReLU se usa como capa porque posee un parámetro alpha entrenable.
- Maxout se implementa como una capa personalizada.
- Softmax se demuestra como salida de clasificación multiclase.
- Identidad/Linear se demuestra como salida típica de regresión.
- Swish también recibe el nombre SiLU.

import numpy as np
import math
#un arreglo ndimensional de numpy simpre lleva un tipo de dato en concreto
x = np.array([1.0,1.2,1.3],np.float32)

#Objetos vectoriales
"""
objeto      descripcion                         uso en IA
Escalar     un unico numero                     Perdidas, tasas de aprendizaje, sesgos, etc
Vercor      Arreglo unidimensional              Muestras, salidas de una capa dentro del modelo
Matriz      Arreglo Bidimensional               Vector de caracteristicas, matriz de pesos, lote de entradas
Tensor      Arreglos de 3 o mas dimensiones     Lote de imagenes, mapas de activacion

"""
#Neurona artificial
"""
La forma mas simple de una neurona artificial recibe un vector de entrada (x), 
un vector de pesos (w) y un sesgo escalar (b) 

---matematicas
Y = mX + b
--- IA
y = wX +b

Despues, si el modelo requiere no linealidad, aplica una funcion de activacion
a = f(y)

"""
x = np.array([1.0,1.2,1.3],np.float32)
w = np.array([0.1,0.2,0.3],np.float32)
b = 0.1
y = np.dot(x,w)+b

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
a = sigmoid(y)


"""
Producto punto

El producto punto entre dos vectores del mismo tamanio produce un escalar 
x = [x1,x2,x3]
y = [y1,y2,y3]
x*y = [x1*y1,x2*y2,x3*y3,...,xn*yn]
"""

#Capa densa para una muestra
"""
Una capa densa con (m) neuronas recibe una entrada de dimension (d) y produce un vector de salida de dimension (m)
Para una sola muestra, si (x) tiene forma (d,) , entonces (w) debe tener forma (d,m) y (b) debe tener forma (m,)

Elemento    Dimension esperada    Interpretacion
x           (d,)                  Una sola muestra
X           (N,d)                 Lote de N muestras
W           (d,m)                 Pesos de una capa con m neuronas
b           (m,)                  Sesgos de una capa con m neuronas
y = X@w + b (n,m)                 Salida lineal para el conjunto de muestras


"""
#Funciones de activacion
"""
- ReLU se define como max(0,x)
- Leaky ReLU se define como max(0.01x,x)
- Tanh se define como (exp(x)-exp(-x))/(exp(x)+exp(-x))
- Sigmoid se define como 1/(1+exp(-x) 
- Softmax se define como exp(x) / sum(exp(x))

"""
#Entrenamiento - Inferencia
"""
Entrenamiento: durante el entrenamiento se modifican los pesos con base a una funcion de perdida
Inferencia: Usar los pesos obtenidos en el entrenamiento para generar una prediccion(salida)
"""
#data set base
X = np.array([
[2.4389536,  3.7051098,  2.2308214,  4.267956  ],
[1.4347757,  4.1926084,  4.0977106,  1.9148788 ],
[1.5340875 , 1.3734006 , 1.1651576 , 3.6456237 ],
[0.85910684, 1.2119873 , 1.7056855 , 3.9947617 ],
[3.4076545  ,2.4863896 , 2.743889,   2.3233895 ],
[1.612485,   2.5630367 , 1.5892828 , 0.00726139],
[2.6862147,  1.7001143 , 3.1898422 , 2.369034  ],
[3.9358408,  0.1861155 , 4.7810326,  0.5449482 ],
[1.351367 ,  1.0466138 , 2.572242,   4.7578664 ]
], dtype = np.float32	)
# clases
y = np.array([1,1,2,2,0,0,1,2,0], dtype = np.int64)
# mapa de clases
class_names = {
    0:'critical',
    1:'warning',
    2:'normal'
}
#exploracion de la matriz de entrada
print("X: ", X)
print("X shape: ", X.shape)
print("X dtype: ", X.dtype)
print("y: ", y)
print("y shape: ", y.shape)
print("y dtype: ", y.dtype)
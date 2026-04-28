import numpy as np
import time
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
y_real = np.array([1,1,2,2,0,0,1,2,0], dtype = np.int64)
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

#Seleccion de la primera muestra
print("primera muestra: ", X[0])
print("Dimension de la primera muestra: ", X[0].shape)
#Seleccion de un columna
feature0 = X[:,0]
print("Primera columna: ", feature0)
#Operaciones elemento a elemento
# print(X*2)
# print(X+2)
#estadisticas por columna indicar el axis
print("media: ", np.mean(X,axis=0))
print("desviacion estandar: ", np.std(X,axis=0))
print("minimo: ", np.min(X,axis=0))
print("maximo: ", np.max(X,axis=0))

#primera neurona artificial
#definir entradas, pesos y sesgo

#entrada
x = X[0]
#pesos
w = np.array([0.5,-0.2,0.8,0.1], dtype=np.float32)
#sesgo
b = 0.3
#calcular la salida lineal
y = np.dot(x,w) + b
print("salida lineal dot: ", y)
y2 = x@w + b
print("salida matricial lineal: ", y2)

#Funcion de neurona artificial
def neuron_forward(x,w,b):
    return np.dot(x,w) + b

y = neuron_forward(x,w,b)
print("salida de la neurona: ", y)

#funciones de activacion
test_values = np.array([-2.0,-0.5,1.2,3.4])
#relu quita los negativos
def reLu(x):
    return np.maximum(0,x)

print("reLu: ", reLu(test_values))

#sigmoide salidas acotadas entre 0 y 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
print("sigmoide: ", sigmoid(test_values))

#softmax cada uno de los datos es la resta del maximo de todos al final los datos suman 1 meneja la probabilidad del valor con respecto a la clase
def softmax(x):
    x_shifted = x - np.max(x)
    exp_values = np.exp(x_shifted)
    return exp_values / np.sum(exp_values)
print("softmax: ", softmax(test_values))
print("softmax suma: ", np.sum(softmax(test_values)))

#Neurona con activacion
def neurona_forward_with_activation(x,w,b, activation = None):
    y = np.dot(x,w) + b
    if activation is None:
        return y
    return activation(y)
# esto se conoce como una capa simple
print("neurona sin activacion: ", neurona_forward_with_activation(x,w,b))
print("neurona con activacion relu: ", neurona_forward_with_activation(x,w,b,reLu))
print("neurona con activacion sigmoid: ", neurona_forward_with_activation(x,w,b,sigmoid))
print("neurona con activacion: softmax", neurona_forward_with_activation(x,w,b,softmax))
# Pasar de una neurona en una capa simple a una capa densa
w1 = np.array([
    [0.5,-0.2,0.8],
    [0.2,0.2,0.8],
    [0.3,-0.5,0.8],
    [0.1,-0.3,0.8],
])
b1 = np.array([0.1, -0.2,0.05])
x = X[0]
y1 = x@w1 + b1
print("resultado y1: ", y1)
print("Dimensiones de y1: ", y1.shape)

y1 = X@w1+b1
print("resultado y1: ", y1)
print("Dimensiones de y1: ", y1.shape)

A1 = reLu(y1)
print("resultado Relu(A1): ", A1)
print("Dimensiones de Relu(A1): ", A1.shape)
#funcion general para todas las muestras
"""
Recibe X = vector
w= neuronas
b= sesgos
activation = funcion de activacion
"""
def dense_forward_batch(X,w,b, activation = None):
    y = X@w + b
    if activation is None:
        return y
    return activation(y)

A2 = dense_forward_batch(X,w1,b1,reLu)
print("resultado Relu(A2): ", A2)
print("Dimensiones de Relu(A2): ", A2.shape)

#construir una red neuronal de dos capas
#capa 1 
#5 neuronas
w1 = np.array([
    [0.2,-0.1,0.5,0.7,-0.3],
    [0.5,0.3,-0.2,0.1,0.6],
    [-0.4,0.8,0.5,-0.6,0.2],
    [0.1,-0.5,0.3,0.2,0.4],
], np.float32)
b1 = np.array([
    0.1,-0.2,0.05, 0,0.15
], np.float32)
#como la w1 saca 5 salidas necesitamos tener un w2 de 5 columnas
w2 =np.array([
    [0.3, -0.2, 0.5],
    [-0.4, 0.6, 0.1],
    [0.2, 0.1, -0.3],
    [0.5, -0.4, 0.2],
    [-0.1, 0.3, 0.4],
],np.float32)

b2 = np.array([0.05, -0.1,0.2 ], np.float32)
print("-"*40,'\n')
A1 = dense_forward_batch(X,w1,b1,reLu)
Z2 = dense_forward_batch(A1,w2,b2)
print("A1: ",A1)
print("-"*40,'\n')
print("Dimensiones salida primera capa:", A1.shape)
print(Z2)
print("-"*40,'\n')
print("Dimensiones salida segunda capa:", Z2.shape)
#probabilidades con softmax por fila

def softmax_batch(X):
    X_shifed = X - np.max(X, axis=1, keepdims=True)
    exp_values = np.exp(X_shifed)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

Prob = softmax_batch(Z2)
print("probabilidades: ", Prob)
print("dimensiones de probabilidades: ", Prob.shape)
print("Suma por filas: ", np.sum(Prob, axis=1))

#obtener la clase predicha
#armax selecciona el indice de la fila con valor maximo
y_pred = np.argmax(Prob, axis=1)
print("clases predichas: ", y_pred)
predicted_classes = [class_names[idx] for idx in y_pred ]
print("clases predichas: ", predicted_classes)

#Funcion completa de inferencia para una red de dos capas
# forward singnifica que todo valor de salida lo pasa a la siguiente capa
def neural_network_forward(X,w1,b1,w2,b2):
    A1 = dense_forward_batch(X,w1,b1,reLu)
    Z2 = dense_forward_batch(A1,w2,b2)
    Prob = softmax_batch(Z2)
    return {
        "A1":A1,
        "Z2":Z2,
        "Prob":Prob,
        "y_pred":np.argmax(Prob, axis=1)
    }
Results = neural_network_forward(X,w1,b1,w2,b2)
print("Resultado A1: ", Results['A1'])
print("Resultado Z2: ", Results['Z2'])
print("Resultado Prob: ", Results['Prob'])
print("Resultado y_pred: ", Results['y_pred'])
print("clases predichas: ", [class_names[idx] for idx in Results["y_pred"]])

#Evaluacion simple de prediccions
# con esto encontramos la presicion
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

print("Valor real ", y_real )
print("Valor predicho ", Results["y_pred"])
acc = accuracy_score(y_real, Results["y_pred"])
print("Accuracy: ", acc)


#Version iterativa de una capa densa
def dense_forward_iterative(X,w,b, activation = None):
    outputs = []
    for i in range(len(X)):
        x = X[i]
        z = X@w + b
        if activation is None:
            outputs.append(z)
        else:
            outputs.append(activation(z))
    return np.array(outputs)
start_time=time.time()
A1_iter = dense_forward_iterative(X,w1,b1,reLu)
end_time = time.time()
start_time2=time.time()
A1_vect = dense_forward_batch(X,w1,b1,reLu)
end_time2 = time.time()

print("Igualdad numerica: ", np.allclose(A1_iter,A1_vect))
print(f"Tiempos iterativo: {end_time - start_time:.10f}") #poner formato no cientificoponer formato no cientifico
print(f"Tiempos vectorizado: {end_time2 - start_time2:.10f}")
# print("Valores A1_iter", A1_iter)
# print("Valores A1_vect", A1_vect)
#evaluar el tiempo 1000 veces
start = time.time()
for i in range(1000):
    A1_iter = dense_forward_iterative(X,w1,b1,reLu)
end_iter = time.time()
start2 = time.time()
for i in range(1000):
    A1_vect = dense_forward_batch(X,w1,b1,reLu)
end_vect = time.time()
print(f"Tiempos iterativo: {end_iter - start:.10f}") 
print(f"Tiempos vectorizado: {end_vect - start2:.10f}")
#inspeccion de las dimensiones
def print_shape(name, array):
    print(f"{name} shape: {array.shape}")

print_shape("X", X)
print_shape("w1", w1)
print_shape("b1", b1)
print_shape("w2", w2)
print_shape("b2", b2)


n_samples, n_features = X.shape
n_features_W1, n_outpues = w1.shape

# Validacion preventiva
def validate_dense_shapes(X,W,b):
    if X.ndim != 2:
        raise ValueError(f"X debe ser una matriz bidimensional {n_samples}x{n_features}")
    if W.ndim != 2:
        raise ValueError(f"W debe ser una matriz bidimensional {n_features_W1}x{n_outpues}")
    if b.ndim != 1:
        raise ValueError("b debe ser un vector unidimensional")
    if X.shape[1] != W.shape[0]:
        raise ValueError("El numero de columnas de X debe ser igual al numero de filas de W")
    if W.shape[1] != b.shape[0]:
        raise ValueError("El numero de columnas de W debe ser igual al numero de filas de b")
    
validate_dense_shapes(X,w1,b1)
# validate_dense_shapes(w1,w2,b2)
def dense_forward_single(x,w,b, activation = None):
    z = np.dot(x,w) + b
    if activation is None:
        return z
    return activation(z)
#prediccion de una muestra
def predict_single(x,w1,b1,w2,b2):
    a1 = dense_forward_single(x,w1,b1,reLu)
    z2 = dense_forward_single(a1,w2,b2)
    p = softmax(z2)
    y_pred = np.argmax(p)
    return {
        "hidden":a1,
        "scores":z2,
        "probabilities":p,
        "predicted_class_index":y_pred,
        "predicted_class_name": class_names[y_pred]
    }
result_single = predict_single(X[0],w1,b1,w2,b2)
print(result_single)
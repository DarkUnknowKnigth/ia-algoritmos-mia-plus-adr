import numpy as np
x = np.array([
  [7,10,7,2],
  [10,12,10,5],
  [6,10,6,5],
  [5,5,5,2],
  [8,10,7,2],
  [10,12,10,5],
  [6,10,6,5],
  [5,5,5,2],
  [10,12,10,5]
],np.float32)
y = np.array([1,0,1,2,1,0,1,2,0], np.float32)
class_names = {0:'critical', 1:'warning', 2:'normal'}
name_to_class = {v:k for k,v in class_names.items()}
#Mostrando lo parametros iniciales de X,Y,Clases y decodificador de clases
print("Dataset x:\n",x,"\nVector Y:\n" ,y,'\nClases:\n',class_names,"\nDecodificador:\n",name_to_class)
#Explorar y validar formas
print("X shape: ", x.shape)
print("Y shape: ", y.shape)
print("Compatibilidad de dimensiones: ", x.shape[0] == y.shape[0])


# explorar una muestra aleatoria [de 0 hasta longitud de la matriz (X)]
index_aleatoreo = np.random.randint(0,x.shape[0])
print(f"Muestra[{index_aleatoreo+1}]: \nX: {x[index_aleatoreo]}\nY: {y[index_aleatoreo]}\nClase: {class_names[y[index_aleatoreo]]}")

#funciones de activacion
#reLu
def reLu(z):
    return np.maximum(0,z)

def softmax(logits):
    """
    Función Softmax: Convierte las salidas puras (logits) en probabilidades.
    Funciona tanto para un solo vector (1D) como para un lote de vectores (2D).
    """
    if logits.ndim == 1:
        # Para un solo vector (inferencia de una muestra)
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)
    # Para un lote de vectores (matriz 2D, entrenamiento o inferencia de lote)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#probando funciones
print("reLu: ", reLu(x[index_aleatoreo]))
print("softmax: ", softmax(x[index_aleatoreo]))
print("sigmoid: ", sigmoid(x[index_aleatoreo]))

# --- Verificación de Funciones de Activación (Punto 3 del Protocolo) ---
print("\n--- Verificación de Funciones de Activación ---")
test_values = np.array([-2.0, -0.5, 0.0, 1.2, 3.4])
relu_output = reLu(test_values)
sigmoid_output = sigmoid(test_values)
softmax_output = softmax(test_values)

print(f"Valores de prueba: {test_values}")
print(f"Salida ReLU: {relu_output}")
print(f"¿ReLU >= 0?: {np.all(relu_output >= 0)}")

print(f"Salida Sigmoid: {sigmoid_output}")
print(f"¿Sigmoid entre 0 y 1?: {np.all((sigmoid_output > 0) & (sigmoid_output < 1))}")

print(f"Salida Softmax: {softmax_output}")
print(f"Suma de Softmax: {np.sum(softmax_output):.4f}")
print(f"¿Suma de Softmax es 1?: {np.isclose(np.sum(softmax_output), 1.0)}")
print("-------------------------------------------------")

#muestra a trabajar 
muestra_x = x[index_aleatoreo]
muestra_y = y[index_aleatoreo]


#validacion de formas de matrices
def validate_dense_shapes(X,W,b):
    n_samples, n_features = X.shape
    n_features_W1, n_outpues = W.shape
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



# declarando los pesos 
w1 = np.array([
    [0.2,-0.1,0.5],
    [0.5,0.3,-0.2],
    [-0.4,0.8,0.5],
    [0.1,-0.5,0.3],
], np.float32)

w2 = np.array([
    [0.3, -0.2, 0.5],
    [-0.4, 0.6, 0.1],
    [0.2, 0.1, -0.3]
  ],np.float32)

print("w1 shape: ", w1.shape)
print("w2 shape: ", w2.shape)
# declarando el sesgo
b1 = np.array([0.1, -0.2, 0.05], np.float32)
b2 = np.array([0.05, -0.1, 0.2], np.float32)
print("b1 shape: ", b1.shape)
print("b2 shape: ", b2.shape)
#funcion para neurona con activacion toma x y realiza el producto punto con w luego suma el sesgo 
def neurona_forward_with_activation(x,w,b, activation = None):
    y = np.dot(x,w) + b
    if activation is None:
        return y
    else:
        return activation(y)
# Calculando la salida de la primera capa (z1) para la muestra seleccionada
z1 = neurona_forward_with_activation(muestra_x,w1,b1)
print("z1 shape: ", z1.shape)

#vectorizar la capa de todo el lote
def dense_forward_batch(X,w,b, activation = None):
    validate_dense_shapes(X, w, b)
    y = X@w + b
    if activation is None:
        return y
    return activation(y)

#primera salida de toda la matriz 
Z1 = dense_forward_batch(x,w1,b1)
print("Z1 shape: ", Z1.shape)

# Armamos la segunda capa segunda capa aplicando una activacion a la capa 1 y luego ejecutamos los calculos con los pesos
A1 = reLu(Z1)
#obtenemos la capa de logits
logits = dense_forward_batch(A1,w2,b2)
print("logits shape: ", logits.shape)
#transformamos los logits en probabilidades
probabilities = softmax(logits)
print("probabilities shape: ", probabilities.shape)
print("probabilities (primeras 5): \n", probabilities[:5])
print("Suma de probabilidades por fila (debe ser 1): ", probabilities.sum(axis=1))
#obtenemos la clase predicha
y_pred = np.argmax(probabilities, axis=1)
print("y_pred: ", y_pred)

# --- Comparación de Vectorización (Punto 5 del Protocolo) ---
import time

def dense_forward_iterative(X, w, b, activation=None):
    """Versión iterativa (lenta) de una capa densa."""
    outputs = []
    for x_sample in X: # Iterar sobre cada muestra
        z = np.dot(x_sample, w) + b
        if activation is None:
            outputs.append(z)
        else:
            outputs.append(activation(z))
    return np.array(outputs)

print("\n--- Comparando Rendimiento: Iterativo vs. Vectorizado ---")

# Medir tiempo de la versión iterativa
start_iter = time.perf_counter()
for _ in range(1000):
    Z1_iter = dense_forward_iterative(x, w1, b1)
end_iter = time.perf_counter()
time_iter = end_iter - start_iter

# Medir tiempo de la versión vectorizada
start_vect = time.perf_counter()
for _ in range(1000):
    Z1_vect = dense_forward_batch(x, w1, b1)
end_vect = time.perf_counter()
time_vect = end_vect - start_vect

print(f"Resultados numéricamente iguales: {np.allclose(Z1_iter, Z1_vect)}")
print(f"Tiempo iterativo (1000 ejecuciones): {time_iter:.6f} segundos")
print(f"Tiempo vectorizado (1000 ejecuciones): {time_vect:.6f} segundos")
print(f"Factor de aceleración (Speedup): {time_iter / time_vect:.2f}x más rápido")
print("-------------------------------------------------")

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
def reLu(x):
    return np.maximum(0,x)
#softmax
def softmax(x):
    x_shifted = x - np.max(x)
    exp_values = np.exp(x_shifted)
    return exp_values / np.sum(exp_values)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#probando funciones
print("reLu: ", reLu(x[index_aleatoreo]))
print("softmax: ", softmax(x[index_aleatoreo]))
print("sigmoid: ", sigmoid(x[index_aleatoreo]))

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
print("probabilities: ", probabilities)
#obtenemos la clase predicha
y_pred = np.argmax(probabilities)
print("y_pred: ", y_pred)



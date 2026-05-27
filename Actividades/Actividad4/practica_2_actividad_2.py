import numpy as np
import time

# --- DATOS Y PARÁMETROS INICIALES ---
print("--- DATOS Y PARÁMETROS INICIALES ---")
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
y_true = np.array([1,0,1,2,1,0,1,2,0], np.int64) 

class_names = {0:'critical', 1:'warning', 2:'normal'}
name_to_class = {v:k for k,v in class_names.items()}
# Pesos
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
# Sesgos
b1 = np.array([0.1, -0.2, 0.05], np.float32)
b2 = np.array([0.05, -0.1, 0.2], np.float32)

print("X shape:", x.shape)
print("y_true shape:", y_true.shape)
print("w1 shape:", w1.shape)
print("b1 shape:", b1.shape)
print("w2 shape:", w2.shape)
print("b2 shape:", b2.shape)
print("-" * 50, "\n")


# --- REFACTORIZACIÓN DE FUNCIONES BASE ---
print("--- REFACTORIZACIÓN DE FUNCIONES BASE ---")

def reLu(z):
    """Función de activación ReLU."""
    return np.maximum(0, z)

def softmax(logits):
    """Función Softmax para un solo vector (1D) o un lote (2D)."""
    if logits.ndim == 1:
        # Para un solo vector
        exps = np.exp(logits - np.max(logits))
        return exps / np.sum(exps)
    # Para un lote de vectores (matriz 2D)
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def dense_forward_single(x_sample, w, b, activation=None):
    """Propagación hacia adelante para una ÚNICA muestra (operación vectorial)."""
    z = np.dot(x_sample, w) + b
    if activation:
        return activation(z)
    return z

def dense_forward_batch(X_batch, w, b, activation=None):
    """Propagación hacia adelante para un LOTE de muestras (operación matricial)."""
    z = X_batch @ w + b
    if activation:
        return activation(z)
    return z

# Corrección de la versión iterativa
def dense_forward_iterative(X_batch, w, b, activation=None):
    """Versión iterativa (lenta) de una capa densa."""
    outputs = []
    for x_sample in X_batch: # Iterar sobre cada muestra
        # Usa la función para una sola muestra
        output_sample = dense_forward_single(x_sample, w, b, activation)
        outputs.append(output_sample)
    return np.array(outputs)

print("-" * 50, "\n")


# --- CONSTRUCCIÓN DEL PIPELINE DE INFERENCIA DOBLE ---
print("--- PIPELINES DE INFERENCIA: ITERATIVO VS. VECTORIZADO ---")

def neural_network_forward_iterative(X, w1, b1, w2, b2):
    """Pipeline de inferencia completo, procesando el dataset muestra por muestra."""
    all_probs = []
    for x_sample in X:
        # Capa 1
        a1 = dense_forward_single(x_sample, w1, b1, activation=reLu)
        # Capa 2 (logits)
        logits = dense_forward_single(a1, w2, b2)
        # Probabilidades para la muestra actual
        probs = softmax(logits)
        all_probs.append(probs)
    return np.array(all_probs)

def neural_network_forward_vectorized(X, w1, b1, w2, b2):
    """Pipeline de inferencia completo, procesando todo el lote con operaciones matriciales."""
    # Capa 1
    A1 = dense_forward_batch(X, w1, b1, activation=reLu)
    # Capa 2 (logits)
    logits = dense_forward_batch(A1, w2, b2)
    # Probabilidades para todo el lote
    probabilities = softmax(logits)
    return probabilities

# Ejecución y validación de equivalencia
probs_iter = neural_network_forward_iterative(x, w1, b1, w2, b2)
probs_vect = neural_network_forward_vectorized(x, w1, b1, w2, b2)

print(f"Forma de salida (iterativa): {probs_iter.shape}")
print(f"Forma de salida (vectorizada): {probs_vect.shape}")
print(f"¿Son las salidas numéricamente equivalentes? -> {np.allclose(probs_iter, probs_vect)}")
print("-" * 50, "\n")


# --- MEDICIÓN DE TIEMPOS Y ESCALABILIDAD ---
print("--- COMPARACIÓN DE RENDIMIENTO (1000 ejecuciones) ---")

# Medir tiempo de la versión iterativa
start_iter = time.perf_counter()
for _ in range(1000):
    _ = neural_network_forward_iterative(x, w1, b1, w2, b2)
end_iter = time.perf_counter()
time_iter = end_iter - start_iter

# Medir tiempo de la versión vectorizada
start_vect = time.perf_counter()
for _ in range(1000):
    _ = neural_network_forward_vectorized(x, w1, b1, w2, b2)
end_vect = time.perf_counter()
time_vect = end_vect - start_vect

print(f"Tiempo total (iterativo):   {time_iter:.6f} segundos")
print(f"Tiempo total (vectorizado): {time_vect:.6f} segundos")
print(f"Factor de aceleración (vectorizado vs iterativo): {time_iter / time_vect:.2f}x más rápido")
print("-" * 50, "\n")


# --- DIAGNÓSTICO DE ERRORES CONTROLADOS ---
print("--- SIMULACIÓN DE ERRORES DE SHAPE ---")

# Error 1: Peso con forma incorrecta
print("\n--> Error 1: Matriz de pesos (W) con forma incorrecta")
try:
    # X tiene 4 features, W_error tiene 5 filas. Debería fallar.
    W_error = np.random.rand(5, 3).astype(np.float32)
    print(f"Intentando X (shape {x.shape}) @ W_error (shape {W_error.shape})...")
    dense_forward_batch(x, W_error, b1)
except ValueError as e:
    print(f"Error capturado de NumPy: {e}")
    print("Explicación: El error ocurre porque para la multiplicación matricial (X @ W), el número de columnas de X (features) debe ser igual al número de filas de W. Aquí, 4 != 5.")

# Error 2: Sesgo con longitud incorrecta (Broadcasting)
print("\n--> Error 2: Vector de sesgo (b) con longitud incorrecta")
try:
    # La salida de X @ w1 es (9, 3). b_error tiene longitud 5. Debería fallar.
    b_error = np.random.rand(5).astype(np.float32)
    z1 = x @ w1
    print(f"Intentando Z1 (shape {z1.shape}) + b_error (shape {b_error.shape})...")
    result = z1 + b_error
except ValueError as e:
    print(f"Error capturado de NumPy: {e}")
    print("Explicación: El error ocurre por las reglas de 'broadcasting' de NumPy. Al sumar una matriz (9, 3) y un vector (5,), NumPy no puede 'estirar' el vector para que coincida con las columnas de la matriz (3 != 5).")

# Error 3: Softmax sobre eje equivocado
print("\n--> Error 3: Softmax aplicado sobre el eje incorrecto")
try:
    # Obtenemos los logits de la red
    A1 = dense_forward_batch(x, w1, b1, activation=reLu)
    logits = dense_forward_batch(A1, w2, b2)
    
    # Aplicamos softmax sobre el eje 0 (columnas) en lugar del eje 1 (filas)
    print(f"Aplicando softmax a logits (shape {logits.shape}) con axis=0...")
    exps = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    probs_wrong_axis = exps / np.sum(exps, axis=0, keepdims=True)
    
    # Verificamos la suma de la primera fila
    sum_primera_fila = np.sum(probs_wrong_axis[0, :])
    print(f"Suma de probabilidades de la primera muestra (fila 0): {sum_primera_fila:.4f}")
    if not np.isclose(sum_primera_fila, 1.0):
        print("Resultado: La suma de las probabilidades para una muestra individual ya no es 1.0.")
        print("Explicación: Al usar axis=0, las probabilidades se calculan a lo largo de las columnas, no de las filas. Esto significa que cada columna suma 1, pero las filas (que representan las probabilidades de clase para cada muestra) no. Esto arruina la interpretación probabilística para la clasificación multiclase.")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")
print("-" * 50, "\n")


# --- SECCIÓN DE EVIDENCIAS ---
print("--- EVIDENCIAS DE EJECUCIÓN Y PREDICCIONES ---")

# Shapes obligatorios
print("\n--> Trazabilidad de Shapes en el pipeline vectorizado:")
Z1 = x @ w1 + b1
A1 = reLu(Z1)
logits_final = A1 @ w2 + b2
probs_final = softmax(logits_final)

print(f"Shape de X (entrada):      {x.shape}")
print(f"Shape de W1 (pesos capa 1):  {w1.shape}")
print(f"Shape de b1 (sesgo capa 1):  {b1.shape}")
print(f"Shape de Z1 (salida lineal): {Z1.shape}")
print(f"Shape de A1 (activación):    {A1.shape}")
print(f"Shape de W2 (pesos capa 2):  {w2.shape}")
print(f"Shape de b2 (sesgo capa 2):  {b2.shape}")
print(f"Shape de Logits (salida final): {logits_final.shape}")
print(f"Shape de Probs (probabilidades): {probs_final.shape}")

# Tabla de predicciones
print("\n--> Tabla de Predicciones:")
y_pred = np.argmax(probs_final, axis=1)
confidences = np.max(probs_final, axis=1)

print(f"{'ID Muestra':<12} | {'Clase Real':<10} | {'Clase Predicha':<15} | {'Confianza':<10}")
print("-" * 65)
for i in range(len(x)):
    id_muestra = f"Muestra {i+1}"
    clase_real = class_names[y_true[i]]
    clase_predicha = class_names[y_pred[i]]
    confianza = f"{confidences[i]:.4f}"
    print(f"{id_muestra:<12} | {clase_real:<10} | {clase_predicha:<15} | {confianza:<10}")
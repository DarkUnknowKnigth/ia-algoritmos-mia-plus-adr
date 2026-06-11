import numpy as np
import matplotlib.pyplot as plt
import json
# Dataset
with open("dataset_procesado.json", "r") as f:
    dataPipeline = json.load(f)
if dataPipeline is None:
    print("Error al cargar los datos")
base_samples = dataPipeline["samples"]
class_to_index = dataPipeline["class_to_index"]
index_to_class = {idx:label for label,idx in class_to_index.items()} 


#Aumento sintetico controlado
def generate_argumented_dataset(base_samples, n_per_class =60, noise=0.12,seed =11):
    """
    Aumenta el tamaño del dataset generando muestras sintéticas.
    Calcula la media y desviación estándar de cada clase y genera nuevos datos 
    usando una distribución normal alrededor de esa media.  
    """
    rng = np.random.default_rng(seed)
    # Agrupa las características (features) de las muestras por su etiqueta (label).
    grouped = {}
    for sample in base_samples:
        grouped.setdefault(sample["label"],[]).append(sample['features'])
    
    X_list, y_list , ids = [],[],[]
    # Itera sobre cada clase y sus características asociadas.
    for label, feature_rows in grouped.items(): 
        arr = np.array(feature_rows, np.float32)
        # Calcula la media y la desviación estándar para las características de la clase actual.
        mean = arr.mean(axis=0)
        # Asegura que la desviación estándar no sea cero para evitar problemas en la generación.
        std = np.maximum(arr.std( axis=0),0.5)
        # Genera 'n_per_class' muestras sintéticas para la clase actual.
        for i in range(n_per_class):
            # Crea una nueva muestra tomando valores de una distribución normal centrada en la media de la clase.
            # El ruido y la desviación estándar controlan la dispersión de los datos sintéticos.
            synthetic = rng.normal(loc=mean, scale=noise+0  *std)
            X_list.append(synthetic)
            y_list.append(class_to_index[label])
            ids.append(f"{label}_{i:03d}")
    # Convierte las listas a arrays de NumPy para operaciones eficientes.
    X =np.array(X_list, np.float32)
    y = np.array(y_list, np.int64)
    return X,y,ids

# Particion estratificada

def split_indices_stratified(y, train_ratio=0.7, validation_ratio=0.15, seed=42):
    """
    Divide el dataset en Entrenamiento (Train), Validación (Validation) y Prueba (Test).
    'Estratificado' significa que mantiene la misma proporción de clases en todos los subconjuntos. 
    """
    rango = np.random.default_rng(seed)
    train_idx, validation_idx, test_idx = [],[],[]
    # Itera sobre cada clase única para realizar la partición estratificada.
    for cls in np.unique(y):
        # Encuentra todos los índices que pertenecen a la clase actual.
        cls_idx = np.where(y==cls)[0]
        # Mezcla los índices de la clase para asegurar aleatoriedad en la partición.
        rango.shuffle(cls_idx)
        n = len(cls_idx)
        # Calcula el número de muestras para cada conjunto (entrenamiento, validación).
        n_train = int(n*train_ratio)
        n_validation = int(n*validation_ratio)
        # Asigna los índices a los conjuntos correspondientes.
        train_idx.extend(cls_idx[:n_train])
        validation_idx.extend(cls_idx[n_train:n_train+n_validation])
        test_idx.extend(cls_idx[n_train+n_validation:])
    return np.array(train_idx) , np.array(validation_idx), np.array(test_idx)

# One-hot encoding

def one_hot(y, n_clasess):
    """
    Convierte etiquetas enteras (ej. 2) en vectores One-Hot (ej. [0, 0, 1]).
    Esto es necesario porque la red neuronal en su capa de salida entrega un vector de
    probabilidades, y necesitamos compararlo con un formato vectorial.
    """
    Y = np.zeros((len(y), n_clasess), dtype=np.float32)
    # Utiliza indexación avanzada de NumPy para colocar un '1' en la columna correspondiente
    # a la clase de cada muestra. Por ejemplo, si y[i] es 2, Y[i, 2] se convierte en 1.
    Y[np.arange(len(y)), y] = 1.0
    return Y

# Funciones de activacion
def relu(z):
    """
    Función de activación Rectified Linear Unit (ReLU).
    Convierte los valores negativos a 0 y deja los positivos igual. 
    Añade 'no linealidad' a la red, permitiéndole aprender patrones complejos.
    """
    return np.maximum(0,z)
def relu_derivative(Z):
    """Derivada de ReLU: Retorna 1 si z > 0, de lo contrario 0. Se usa en el Backpropagation."""
    return (Z > 0).astype(np.float32)

def softmax(logits):
    """
    Función Softmax: Convierte las salidas puras (logits) de la última capa en probabilidades.
    Todos los valores resultantes estarán entre 0 y 1, y su suma será exactamente 1.
    """
    # Restar el máximo de los logits es un truco para la estabilidad numérica.
    # Evita el desbordamiento (overflow) al calcular el exponente de números muy grandes.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

# Inicializacion de parametros
def initialize_parameters(n_features, n_hidden, n_classes, seed=42):
    """
    Inicializa los pesos (W) y sesgos (b) de la red.
    Una buena inicialización es clave para un entrenamiento efectivo.
    """
    rng = np.random.default_rng(seed)
    # Inicialización de He: recomendada para redes con activación ReLU.
    # Ayuda a prevenir que los gradientes se desvanezcan o exploten durante el entrenamiento.
    w1 = rng.normal(0, np.sqrt(2/n_features), size=(n_features, n_hidden))
    b1 = np.zeros((1, n_hidden))
    w2 = rng.normal(0, np.sqrt(2/n_hidden), size=(n_hidden, n_classes))
    b2 = np.zeros((1, n_classes))
    return {"w1":w1, "b1":b1, "w2":w2, "b2":b2}

# Propagacion hacia adelante
def forward(x, params):
    """
    Realiza el pase de propagación hacia adelante (inferencia).
    Calcula la salida de la red para una entrada 'x' dada.
    """
    w1,b1,w2,b2 = params["w1"], params["b1"], params["w2"], params["b2"]
    # Capa 1: Cálculo lineal seguido de la activación ReLU.
    Z1 = x @ w1 + b1
    A1 = relu(Z1)
    # Capa 2: Cálculo lineal para obtener los logits (salidas antes de softmax).
    logits = A1 @ w2 + b2
    # Aplicar Softmax para convertir los logits en un vector de probabilidades.
    probs = softmax(logits)
    # Guardar valores intermedios en 'cache' para usarlos en la retropropagación.
    cache = {
        'X':x,
        'Z1':Z1,
        'A1':A1,
        'logits':logits,
        'probs':probs
    }
    return probs, cache

# perdida
def cross_entropy_loss(probs, y_true):
    """
    Calcula la pérdida de entropía cruzada.
    Mide qué tan "equivocadas" están las predicciones del modelo en comparación con las etiquetas reales.
    El objetivo del entrenamiento es minimizar esta pérdida.
    """
    n = len(y_true)
    # Selecciona la probabilidad predicha para la clase correcta de cada muestra.
    selected = probs[np.arange(n), y_true]
    # Calcula el logaritmo de esas probabilidades y promedia el resultado.
    # Se suma un valor muy pequeño (1e-9) para evitar calcular log(0), que es infinito.
    return -np.mean(np.log(selected + 1e-9))

# Retropropagacion
def backward(y_true, params, cache):
    """
    Realiza el pase de retropropagación (backpropagation).
    Calcula los gradientes (derivadas) de la pérdida con respecto a cada peso y sesgo.
    Estos gradientes indican cómo ajustar los parámetros para reducir la pérdida.
    """
    # Recupera los valores intermedios del pase hacia adelante.
    X = cache['X']
    A1 = cache['A1']
    Z1 = cache['Z1']
    probs = cache['probs']
    w2 = params['w2']
    n = X.shape[0]

    # --- Gradientes de la Capa de Salida (Capa 2) ---
    # El gradiente de la pérdida de entropía cruzada con softmax es simplemente (probs - y_one_hot).
    # Aquí se calcula de forma eficiente: se restan 1 a las probabilidades de la clase correcta.
    dlogits = probs.copy()
    dlogits[np.arange(n), y_true] -=  1
    # Se promedia el gradiente sobre el número de muestras.
    dlogits /= n
    # Gradiente para los pesos w2: es el producto de la activación de la capa anterior (A1) y el gradiente de los logits.
    dw2 = A1.T @ dlogits
    # Gradiente para el sesgo b2: es la suma de los gradientes de los logits.
    db2 = np.sum(dlogits, axis=0, keepdims=True)

    # --- Retropropagar el error a la Capa Oculta (Capa 1) ---
    # Gradiente con respecto a la activación A1.
    da1 = dlogits @ w2.T
    # Gradiente con respecto a la salida lineal Z1, aplicando la derivada de ReLU.
    # El gradiente es cero donde Z1 era negativo.
    dz1 = da1 * relu_derivative(Z1)
    # Gradiente para los pesos w1.
    dw1 = X.T @ dz1
    # Gradiente para el sesgo b1.
    db1 = np.sum(dz1,axis=0, keepdims=True)
    return {
        'dw2':dw2,
        'db2':db2,
        'dw1':dw1,
        'db1':db1
    }

# verificacion numerica de gradientes
def numerical_gradient_check(x_small, y_small, params,param_name="w1", index= (0,0), epsilon=1e-7):
    """
    Verifica si el cálculo del gradiente analítico (backward) es correcto comparándolo
    con una aproximación numérica. Es una herramienta de depuración muy útil.
    """
    params_plus = {key: value.copy() for key, value in params.items()}
    params_minus = {key: value.copy() for key, value in params.items()}
    # Perturba un solo parámetro (peso o sesgo) en una cantidad muy pequeña (epsilon).
    params_plus[param_name][index] += epsilon
    params_minus[param_name][index] -= epsilon
    # Calcula la pérdida con el parámetro perturbado hacia arriba y hacia abajo.
    loss_plus = cross_entropy_loss(forward(x_small, params_plus)[0], y_small)
    loss_minus = cross_entropy_loss(forward(x_small, params_minus)[0], y_small)
    # Aproxima el gradiente usando la fórmula de la diferencia central.
    numerical = (loss_plus - loss_minus) / (2 * epsilon)
    # Obtiene el gradiente calculado por la función backward (analítico).
    probs, cache = forward(x_small, params)
    analytic = backward(y_small, params, cache)['d' + param_name][index]
    return numerical, analytic, abs(numerical - analytic)
# Actualizacion y metricas
def update_parameters(params, grads, lerning_rate):
    """Aplica la regla de actualización del descenso de gradiente para ajustar los parámetros."""
    for name in ["w1", "b1", "w2", "b2"]:
        params[name] -= lerning_rate * grads["d" + name]
    return params
def predict(x, params):
    """Realiza una predicción para un conjunto de datos, devolviendo la clase y las probabilidades."""
    probs, _ = forward(x, params)
    return np.argmax(probs, axis=1), probs
def accuracy_score(y_true, y_pred):
    """Calcula la precisión (accuracy) del modelo."""
    return np.mean(y_true == y_pred)
def confusion_matrix_np(y_true, y_pred, n_classes):
    """Construye una matriz de confusión para evaluar el rendimiento por clase."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm

# Entrenamiento completo
def train_network(X_train, y_train, X_val, y_val, n_hidden=8, lr=0,epochs=600, seed=42):
    """
    Orquesta el ciclo de entrenamiento completo de la red neuronal.
    """
    # 1. Inicializa los parámetros de la red.
    params = initialize_parameters(X_train.shape[1], n_hidden, len(np.unique(y_train)), seed)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    # 2. Itera a través del número especificado de épocas (ciclos de entrenamiento).
    for epoch in range(epochs):
        # 3. Realiza el pase hacia adelante para obtener predicciones y el cache.
        probs, cache = forward(X_train, params)
        # 4. Calcula la pérdida de entrenamiento.
        loss = cross_entropy_loss(probs, y_train)
        # 5. Realiza el pase hacia atrás para obtener los gradientes.
        grads = backward(y_train, params, cache)
        # 6. Actualiza los parámetros usando los gradientes y la tasa de aprendizaje.
        params = update_parameters(params, grads, lr)
        
        # 7. Evalúa el modelo en los datos de entrenamiento y validación para monitorear el progreso.
        y_pred_train, train_probs = predict(X_train, params)
        y_pred_val, val_probs = predict(X_val, params)
        val_loss = cross_entropy_loss(val_probs, y_val)

        # 8. Guarda las métricas en el historial.
        history["train_loss"].append(loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(accuracy_score(y_train, y_pred_train))
        history["val_acc"].append(accuracy_score(y_val, y_pred_val))
        
        # Imprime el progreso cada 100 épocas.
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: train_loss = {loss:.4f}, val_loss = {val_loss:.4f}")   
    return params, history
# Reporte de errores
def error_report(ids_subset, y_true, y_pred, probs, index_to_class):
    """Genera un reporte detallado de las predicciones incorrectas."""
    report = []
    for sample_id, true_label, pred_label, prob in zip(ids_subset, y_true, y_pred, probs):
        report.append(  {
            "id": sample_id,
            "true_label": index_to_class[int(true_label)],
            "pred_label": index_to_class[int(pred_label)],
            "acurracy_pred": float(np.max(prob)),
            "probabilities": np.round(prob, 4),
        })
    return report
# Guardado, carga e inferencia nueva
def save_parameters(params,mean,std, filename="modelo.npz"):
    """
    Guarda los parámetros entrenados del modelo (pesos, sesgos) y las estadísticas
    de normalización (media, desviación estándar) en un archivo.
    """
    np.savez(
        filename, 
        W1=params["w1"],
        b1=params["b1"],
        W2=params["w2"],
        b2=params["b2"],
        mean=mean,
        std=std
    )
        
def load_parameters(filename="modelo.npz"):
    """Carga los parámetros y estadísticas de normalización desde un archivo."""
    data = np.load(filename)
    params = {
        "w1": data["W1"],
        "b1": data["b1"],
        "w2": data["W2"],
        "b2": data["b2"]
    }
    mean = data["mean"]
    std = data["std"]
    return params, mean, std
def predict_single(features, params, mean, std):
    """
    Realiza una predicción para una única muestra nueva.
    Es crucial normalizar la nueva muestra usando la media y std del conjunto de entrenamiento.
    """
    # Convierte la lista de características en un array de NumPy con la forma correcta (1, n_features).
    x = np.array(features, np.float32).reshape(1, -1)
    # Normaliza la muestra usando la media y desviación estándar del entrenamiento.
    x_normalized = (x - mean) / std
    # Realiza la inferencia (pase hacia adelante).
    probs, _ = forward(x_normalized, params)
    # Obtiene el índice de la clase con la probabilidad más alta.
    pred_idx = int(probs.argmax(axis=1)[0])
    return index_to_class[pred_idx], probs[0]

# Graficas
def plot_history(history, experiment_name="experiment"):
    """Visualiza las curvas de pérdida y precisión durante el entrenamiento."""
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")
    plt.title("Evolution of loss during training - "+experiment_name)
    plt.legend()
    plt.tight_layout()
    curve_name_loss = experiment_name+"-loss_curve.png"
    plt.savefig(curve_name_loss, dpi=150)
    plt.show()

    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Evolution of accuracy during training - "+experiment_name)
    plt.legend()
    plt.tight_layout()
    curve_name_accuracy = experiment_name+"-accuracy_curve.png"
    plt.savefig(curve_name_accuracy, dpi=150)
    plt.show()
    return curve_name_loss, curve_name_accuracy

# Laboratorio de experimientos
def run_experiment(experiment = {"n_hidden": 4, "lr":0.01, "epochs": 300, "seed": 42}, experiment_id=0, X_train_n=None, y_train=None, X_validation_n=None, y_validation=None):
    params_trained, history = train_network(
        X_train_n, 
        y_train, 
        X_validation_n, 
        y_validation,
        n_hidden=experiment["n_hidden"], 
        lr=experiment["lr"], 
        epochs=experiment["epochs"], 
        seed=experiment["seed"]
    )
    y_pred_val, _ = predict(X_validation_n, params_trained)
    metrics = {
        "n_hidden": experiment["n_hidden"],
        "lr": experiment["lr"],
        "val_acc": accuracy_score(y_validation, y_pred_val),
        "val_loss": history["val_loss"][-1],
        "confussion_matrix": confusion_matrix_np(y_validation, y_pred_val, len(class_to_index)).tolist()
    }
    print(metrics)
    # 10. Guardado del mejor modelo y prueba de inferencia en una muestra nueva
    name = f"modelo-{experiment_id:03d}.npz"
    return name, metrics, params_trained , history

# Programa principal
def main():
    print("---- Practiva 3: Red neuronal desde cero ----")
    # 1. Generación del dataset sintético
    X, y, ids = generate_argumented_dataset(base_samples, n_per_class=60, noise=0.12, seed=42)
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    print("Distribution: ", {index_to_class[k]: int(np.sum(y==k)) for k in np.unique(y)})
    # 2. Partición estratificada en conjuntos de entrenamiento, validación y prueba
    train_idx, validation_idx, test_idx = split_indices_stratified(y, seed=42)
    X_train, y_train = X[train_idx], y[train_idx]
    X_validation, y_validation = X[validation_idx], y[validation_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # 3. Normalización (Standard Scaler)
    # Se calcula la media y std SOLO en el conjunto de entrenamiento para evitar fuga de datos.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    # Se aplica la misma transformación a todos los conjuntos.
    X_train_n = (X_train - mean) / std
    X_validation_n = (X_validation - mean) / std
    X_test_n = (X_test - mean) / std

    # errors = error_report([ids[i] for i in test_idx], y_test, y_pred_test, probs_test, index_to_class)
    # print("Errores detectados:", len(errors))
    # for item in errors[:5]:
    #     print(item)
    
    # 9. Experimentos guiados para encontrar buenos hiperparámetros (A,B,C,D)
    #seed aleatorea para los experimentos
    random_seed =np.random.randint(1000)
    experiments = [
        {"n_hidden": 8, "lr":0.003, "epochs": 600, "seed": random_seed},
        {"n_hidden": 8, "lr":0.03, "epochs": 600, "seed": random_seed},
        {"n_hidden": 8, "lr":0.3, "epochs": 600, "seed": random_seed},
        {"n_hidden": 16, "lr":0.03, "epochs": 600, "seed": random_seed},
    ]
    results = []
    print("\n --- Experimentos guiados")
    #Muestra aleatoria
    random_index = np.random.randint(len(X_test))
    X_test_to_predict = X_test[random_index]
    y_test_to_predict = y_test[random_index]

    for index,experiment in enumerate(experiments):
        print("\nExperimento:")
        print(experiment)
        # Usando la funcion de correr experimentos y catalogandolos
        experiment_name, metrics, params_trained, history = run_experiment(experiment,index, X_train_n, y_train, X_validation_n, y_validation)
        #Guardar los nuevos experimentos con su nombre del archivo
        save_parameters(params_trained, mean, std, filename=experiment_name)
        #Cargar el modelo creado y predecir una muestra 
        params_loaded, mean_loaded, std_loaded = load_parameters(experiment_name)
        #predecir la muestra aleatorea de test
        label, prob = predict_single(X_test_to_predict, params_loaded, mean_loaded, std_loaded)
        print("\n Inferencia de muestra nueva:")
        print("Clase predicha: ", label)
        print("Probabilidades: ", np.round(prob, 4))
        # 11. Visualización de los resultados del entrenamiento
        curve_name_loss, curve_name_accuracy = plot_history(history,experiment_name)
        #guardar los registros del experimento, imagenes de graficas, modelos y los resultados de la prediccion contra lo real
        results.append({
            "metrics":metrics,
            "filename":experiment_name, 
            "experiment":experiment, 
            "curve_name_loss_image":curve_name_loss, 
            "curve_name_accuracy_image":curve_name_accuracy,
            "predicted_label":label,
            "real_label":index_to_class[int(y_test_to_predict)],
            "probabilities":np.round(prob, 4).tolist()
        })
    #resumen de resultados (guardar experimentos y ubicaciones de los modelos en un json)
    json_object = json.dumps(results, indent=4)
    with open("results.json", "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    main()

#inferencia: consiste en usar parametros ya existentes para transformar una entrada en una salida
#Entrenamiento: Modificar esos parametros con base en una senal de error
import numpy as np
#graficas
import matplotlib.pyplot as plt
import json

#data set base
base_samples = [
    {
        "id": "sample_001", #identificador debe ser unico
        "features": [0.45, 1.12, 2.89, 0.76], #valores de medicion
        "label": "normal", #etiqueta
        "metadata": { #datos adicionales con los que no se realizan calculos
            "source": "sensor_a",
            "quality": "high"
        }
    },
    {
        "id": "sample_002",
        "features": [1.24, 5.67, 12.4, 1.2],
        "label": "warning",
        "metadata": {
            "source": "sensor_b",
            "quality": "low"
        }
    },
    {
        "id": "sample_003",
        "features": [0.89, 92.34, 115.1, 88.42],
        "label": "critical",
        "metadata": {
            "source": "sensor_c",
            "quality": "high"
        }
    },
    {
        "id": "sample_004",
        "features": [0.31, 0.98, 3.05, 1.15],
        "label": "normal",
        "metadata": {
            "source": "sensor_a",
            "quality": "high"
        }
    },
    {
        "id": "sample_005",
        "features": [2.15, 8.42, 15.8, 6.92],
        "label": "warning",
        "metadata": {
            "source": "sensor_b",
            "quality": "low"
        }
    },
    {
        "id": "sample_006",
        "features": [1.02, 78.12, 95.5, 102.3],
        "label": "critical",
        "metadata": {
            "source": "sensor_c",
            "quality": "high"
        }
    },
    {
        "id": "sample_007",
        "features": [0.31, 0.98, 3.05, 1.15],
        "label": "normal",
        "metadata": {
            "source": "sensor_a",
            "quality": "high"
        }
    },
    {
        "id": "sample_008",
        "features": [2.15, 8.42, 15.8, 6.92],
        "label": "warning",
        "metadata": {
            "source": "sensor_b",
            "quality": "low"
        }
    },
    {
        "id": "sample_009",
        "features": [1.02, 78.12, 95.5, 102.3],
        "label": "critical",
        "metadata": {
            "source": "sensor_c",
            "quality": "high"
        }
    },
    {
        "id": "sample_010",
        "features": [1.22, 18.12, 65.5, 12.3],
        "label": "critical",
        "metadata": {
            "source": "sensor_c",
            "quality": "high"
        }
    }
]
class_to_index = {
    'critical': 0,
    'normal': 1,
    'warning': 2
}
index_to_class = {idx:label for label,idx in class_to_index.items()} 

# aumentacion de datos: generar datos sinteticos
def generate_argumented_dataset(base_samples, n_per_class =60, noise=0.12,seed =11):
    rng = np.random.default_rng(seed)
    grouped = {}
    for sample in base_samples:
        grouped.setdefault(sample["label"],[]).append(sample['features'])
    x_list, y_list , ids = [],[],[]
    for label, feature_rows in grouped.items(): 
        arr = np.array(feature_rows, np.float32)
        mean = arr.mean(axis=0)
        std = np.maximum(arr.std( axis=0),0.5)
        for i in range(n_per_class):
            synthetic = rng.normal(loc=mean, scale=noise+0.15*std)
            x_list.append(synthetic)
            y_list.append(class_to_index[label])
            ids.append(f"{label}_{i:03d}")
    X =np.array(x_list, np.float32)
    y = np.array(y_list, np.int64)
    return X,y,ids

X, y, ids = generate_argumented_dataset(base_samples)
print("X: ", X)
print("y: ", y)
print("ids: ", ids)
print("Dimension de x", X.shape)
print("Dimension de y", y.shape)
print("Dimension de ids", len(ids ))
print("Distribucion: " ,{index_to_class[k]: int(np.sum(y==k)) for k in np.unique(y)} )

#particion train / validation /test
def split_indices_stratified(y, train_ratio=0.7, validation_ratio=0.15, seed=42):
    rango = np.random.default_rng(seed)
    train_idx, validation_idx, test_idx = [],[],[]
    for cls in np.unique(y):
        cls_idx = np.where(y==cls)[0]
        rango.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(n*train_ratio)
        n_validation = int(n*validation_ratio)
        train_idx.extend(cls_idx[:n_train])
        validation_idx.extend(cls_idx[n_train:n_train+n_validation])
        test_idx.extend(cls_idx[n_train+n_validation:])
    return np.array(train_idx) , np.array(validation_idx), np.array(test_idx)

#division de los datos 
train_idx, validation_idx, test_idx = split_indices_stratified(y, train_ratio=0.7, validation_ratio=0.15, seed=42)

print("train_idx shape: ", train_idx.shape)
print("validation_idx shape: ", validation_idx.shape)
print("test_idx shape: ", test_idx.shape)   

X_train, Y_train = X[train_idx], y[train_idx]
X_validation, Y_validation = X[validation_idx], y[validation_idx]
X_test, Y_test = X[test_idx], y[test_idx]

#Normalizacion 
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std = np.where(std == 0,1.0, std)

X_train_normalized = (X_train - mean) / std
X_validation_normalized = (X_validation - mean) / std
X_test_normalized = (X_test - mean) / std

print("X_train_normalized shape: ", X_train_normalized)
print("X_validation_normalized shape: ", X_validation_normalized)
print("X_test_normalized shape: ", X_test_normalized)

#one-hot encoding
# un vector codigicados con ceros que cambia dependiendo de la clase (ayuda visual)
def one_hot(y, n_clasess):
    Y = np.zeros((len(y), n_clasess), dtype=np.float32)
    Y[np.arange(len(y)), y] = 1.0
    return Y

y_train_one_hot = one_hot(Y_train, len(class_to_index))
y_validation_one_hot = one_hot(Y_validation, len(class_to_index))
y_test_one_hot = one_hot(Y_test, len(class_to_index))

print("y_train_one_hot: ", y_train_one_hot)
print("y_validation_one_hot: ", y_validation_one_hot) 
print("y_test_one_hot: ", y_test_one_hot)

#funciones de activacion
def relu(z):
    return np.maximum(0,z)
def relu_prime(z):
    return (z>0).astype(np.float32)
def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

Z_prueba = np.array([[-2.0,0.0,3.0]])
print("reLU", relu(Z_prueba))
print("reLU_prime: ", relu_prime(Z_prueba))

logits_prueba = np.array([[2.0,1.0,0.1]])
print("softmax: ", softmax(logits_prueba))
print("Suma filas softmax: ", softmax(logits_prueba).sum(axis=1) )

#inicializacion de parametros
def initialize_parameters(n_features, n_hidden, n_classes, seed=42):
    range = np.random.default_rng(seed)
    w1 = range.normal(0, np.sqrt(2/n_features), size=(n_features, n_hidden))
    b1 = np.zeros((1, n_hidden))
    w2 = range.normal(0, np.sqrt(2/n_hidden), size=(n_hidden, n_classes))
    b2 = np.zeros((1, n_classes))
    return {"w1":w1, "b1":b1, "w2":w2, "b2":b2}
params = initialize_parameters( n_features=4, n_hidden=8, n_classes=3, seed=42)

for name, value in params.items():
    print(f"{name}: {value.shape}")

#propagacion hacia adelante
def forward(x, params):
    w1,b1,w2,b2 = params["w1"], params["b1"], params["w2"], params["b2"]
    z1 = x @ w1 + b1
    a1 = relu(z1)
    logits = a1 @ w2 + b2
    probabilities = softmax(logits)
    cache = {
        'X':x,
        'z1':z1,
        'a1':a1,
        'logits':logits,
        'probabilities':probabilities
    }
    return probabilities, cache
# primer prediccion
probs, cache = forward(X_train_normalized[:5], params)
print("probs dimensiones: ", probs.shape)
print("probs ejemplo: ", probs[0])

#segundo paso - calculo de perdidas (cross entropy)
def cross_entropy_lost(probabilities, y_true):
    n = len(y_true)
    selected = probabilities[np.arange(n), y_true]
    return np.mean(np.log(selected + 1e-9))

loss_initial = cross_entropy_lost(probs, Y_train[:5])
print("perdida inicial: ", loss_initial)

#retro propagacion paso a paso

def backward(y_true, params, cache):
    X = cache['X']
    a1 = cache['a1']
    z1 = cache['z1']
    logits = cache['logits']
    probabilities = cache['probabilities']
    w2 = params['w2']
    n = X.shape[0]
    dlogits = probabilities.copy()
    dlogits[np.arange(n), y_true] -= 1.0
    dlogits /= n

    dw2 = a1.T @ dlogits
    db2 = dlogits.sum(axis=0, keepdims=True)
    da1 = dlogits @ w2.T
    dz1 = da1 * relu_prime(z1)
    dw1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)
    return {
        'dw2':dw2,
        'db2':db2,
        'dw1':dw1,
        'db1':db1
    }

grads = backward(Y_train[:5], params, cache)
for name, value in grads.items():
    print(name, value.shape)

#cuarto paso - entrenamiento
def train_step(params, y_true, learning_rate, cache):
    grads = backward(y_true, params, cache)
    
    # Actualización de parámetros usando descenso de gradiente
    params["w1"] -= learning_rate * grads["dw1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["w2"] -= learning_rate * grads["dw2"]
    params["b2"] -= learning_rate * grads["db2"]
    
    return params
    
train = train_step(params, Y_train[:5], 0.01, cache)
for name, value in train.items():
    print(name, value.shape)

#quinto
def train_epoch(params, X_train, y_train):
    probabilities, cache = forward(X_train, params)
    loss = cross_entropy_lost(probabilities, y_train)
    params = train_step(params, y_train, 0.01, cache)
    return params, loss
    
params, loss = train_epoch(params, X_train_normalized, Y_train)
print("loss: ", loss)

#sexto
def train_model(X_train, Y_train, X_val, Y_val, n_hidden=16, epochs=1000, lr=0.01, seed=42):
    n_features = X_train.shape[1]
    n_classes = len(np.unique(Y_train))
    params = initialize_parameters(n_features, n_hidden, n_classes, seed)
    
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        # Entrenamiento
        probs_train, cache_train = forward(X_train, params)
        loss_train = cross_entropy_lost(probs_train, Y_train)
        params = train_step(params, Y_train, lr, cache_train)
        
        # Validación
        probs_val, _ = forward(X_val, params)
        loss_val = cross_entropy_lost(probs_val, Y_val)
        
        history["train_loss"].append(loss_train)
        history["val_loss"].append(loss_val)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: train_loss = {loss_train:.4f}, val_loss = {loss_val:.4f}")
            
    return params, history

# Ejecución del entrenamiento completo
params_final, history = train_model(X_train_normalized, Y_train, X_validation_normalized, Y_validation)

def evaluate_model(params, X_test, Y_test):
    probs, _ = forward(X_test, params)
    loss = cross_entropy_lost(probs, Y_test)
    predictions = probs.argmax(axis=1)
    accuracy = np.mean(predictions == Y_test)
    return loss, accuracy

loss_final, accuracy_final = evaluate_model(params_final, X_test_normalized, Y_test)
print(f"Final Test Loss: {loss_final:.4f}, Final Test Accuracy: {accuracy_final:.4f}")

def graphic(history, X_test_normalized, Y_test):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2,1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Curva de Pérdida")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    # Visualización de predicciones vs reales en el set de prueba
    probs_test, _ = forward(X_test_normalized, params_final)
    y_pred = probs_test.argmax(axis=1)
    plt.scatter(range(len(Y_test)), Y_test, label="Real", alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, label="Predicho", marker='x')
    plt.title("Predicciones vs Real (Test)")
    plt.xlabel("Muestra")
    plt.ylabel("Clase")
    plt.legend()

    plt.tight_layout()
    plt.show()
#graphic()

def export_model(params, mean, std, filename="test_model_params.npz"):
    np.savez(filename, w1=params["w1"], b1=params["b1"], w2=params["w2"], b2=params["b2"], mean=mean, std=std)
    print(f"Modelo exportado exitosamente a {filename}")
export_model(params_final, mean, std)
def export_as_json(params, mean, std, filename="test_model_params.json"):
    model_dict = {
        "w1": params["w1"].tolist(),
        "b1": params["b1"].tolist(),
        "w2": params["w2"].tolist(),
        "b2": params["b2"].tolist(),
        "mean": mean.tolist(),
        "std": std.tolist()
    }
    with open(filename, "w") as f:
        json.dump(model_dict, f)
    print(f"Modelo exportado exitosamente a {filename}")
export_as_json(params_final, mean, std)

def load_model(filename="test_model_params.npz"):
    data = np.load(filename)
    params = {
        "w1": data["w1"],
        "b1": data["b1"],
        "w2": data["w2"],
        "b2": data["b2"]
    }
    mean = np.array(data["mean"])
    std = np.array(data["std"])
    return params, mean, std
params_loaded, mean_loaded, std_loaded = load_model()
print("Parámetros cargados:")
for name, value in params_loaded.items():
    print(f"{name}: {value.shape}")
print("Media cargada:", mean_loaded)
print("Desviación estándar cargada:", std_loaded)


def predict_new_sample(features, filename="test_model_params.npz"):
    # 1. Cargar el modelo y los parámetros de normalización
    params, mean_loaded, std_loaded = load_model(filename)
    
    # 2. Convertir la entrada a un array de NumPy
    x_input = np.array([features], dtype=np.float32)
    
    # 3. NORMALIZACIÓN (CRUCIAL)
    # Debes usar la media y desviación del entrenamiento, no las del nuevo dato
    x_normalized = (x_input - mean_loaded) / std_loaded
    
    # 4. Ejecutar el Forward (Inferencia)
    # Reutilizamos tu función forward
    probabilities, _ = forward(x_normalized, params)
    
    # 5. Obtener la clase con mayor probabilidad
    prediction_index = np.argmax(probabilities)
    class_name = index_to_class[prediction_index]
    confidence = probabilities[0][prediction_index] * 100
    
    return class_name, confidence

# --- EJEMPLO DE USO ---
# Supongamos que recibes una lectura nueva de un sensor
nueva_lectura = [1.5, 20.0, 70.0, 15.0] # Un caso crítico según tus datos

clase, confianza = predict_new_sample(nueva_lectura)

print(f"Resultado de la predicción: {clase}")
print(f"Confianza: {confianza:.2f}%")


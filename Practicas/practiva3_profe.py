import numpy as np
import matplotlib.pyplot as plt

# Dataset
base_samples = [
  {
    "id": "sample_001",
    "features": [0.45, 1.12, 2.89, 0.76],
    "label": "normal",
    "metadata": {
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
      "features": [0.55, 1.25, 2.70, 0.85],
      "label": "normal",
      "metadata": {
        "source": "sensor_a",
        "quality": "high"
      }
  },
  {
      "id": "sample_008",
      "features": [1.80, 6.10, 13.5, 2.1],
      "label": "warning",
      "metadata": {
        "source": "sensor_b",
        "quality": "low"
      }
  },
  {
      "id": "sample_009",
      "features": [1.15, 85.20, 105.4, 95.10],
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


#Aumento sintetico controlado
def generate_argumented_dataset(base_samples, n_per_class =60, noise=0.12,seed =11):
    """
    Aumenta el tamaño del dataset generando muestras sintéticas.
    Calcula la media y desviación estándar de cada clase y genera nuevos datos 
    usando una distribución normal alrededor de esa media.  
    """
    rng = np.random.default_rng(seed)
    grouped = {}
    for sample in base_samples:
        grouped.setdefault(sample["label"],[]).append(sample['features'])
    X_list, y_list , ids = [],[],[]
    for label, feature_rows in grouped.items(): 
        arr = np.array(feature_rows, np.float32)
        mean = arr.mean(axis=0)
        std = np.maximum(arr.std( axis=0),0.5)
        for i in range(n_per_class):
            synthetic = rng.normal(loc=mean, scale=noise+0  *std)
            X_list.append(synthetic)
            y_list.append(class_to_index[label])
            ids.append(f"{label}_{i:03d}")
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

# One-hot encoding

def one_hot(y, n_clasess):
    """
    Convierte etiquetas enteras (ej. 2) en vectores One-Hot (ej. [0, 0, 1]).
    Esto es necesario porque la red neuronal en su capa de salida entrega un vector de
    probabilidades, y necesitamos compararlo con un formato vectorial.
    """
    Y = np.zeros((len(y), n_clasess), dtype=np.float32)
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
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

# Inicializacion de parametros
def initialize_parameters(n_features, n_hidden, n_classes, seed=42):
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, np.sqrt(2/n_features), size=(n_features, n_hidden))
    b1 = np.zeros((1, n_hidden))
    w2 = rng.normal(0, np.sqrt(2/n_hidden), size=(n_hidden, n_classes))
    b2 = np.zeros((1, n_classes))
    return {"w1":w1, "b1":b1, "w2":w2, "b2":b2}

# Propagacion hacia adelante
def forward(x, params): 
    w1,b1,w2,b2 = params["w1"], params["b1"], params["w2"], params["b2"]
    Z1 = x @ w1 + b1
    A1 = relu(Z1)
    logits = A1 @ w2 + b2
    probs = softmax(logits)
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
    n = len(y_true)
    selected = probs[np.arange(n), y_true]
    return np.mean(np.log(selected + 1e-9))

# Retropropagacion
def backward(y_true, params, cache): 
    X = cache['X']
    A1 = cache['A1']
    Z1 = cache['Z1']
    logits = cache['logits']
    probs = cache['probs']
    w2 = params['w2']
    n = X.shape[0]
    dlogits = probs.copy()
    dlogits[np.arange(n), y_true] -=  1
    dlogits /= n
    dw2 = A1.T @ dlogits
    db2 = dlogits.sum(axis=0, keepdims=True)
    da1 = dlogits @ w2.T
    dz1 = da1 * relu_derivative(Z1)
    dw1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)
    return {
        'dw2':dw2,
        'db2':db2,
        'dw1':dw1,
        'db1':db1
    }
# verificacion numerica de gradientes
def numerical_gradient_check(x_small, y_small, params,param_name="w1", index= (0,0), epsilon=1e-7):
    params_plus = {key: value.copy() for key, value in params.items()}
    params_minus = {key: value.copy() for key, value in params.items()}
    params_plus[index[0]][index[1]][index[2]] += epsilon
    params_minus[index[0]][index[1]][index[2]] -= epsilon
    loss_plus = cross_entropy_loss(forward(x_small, params_plus)[0], y_small)
    loss_minus = cross_entropy_loss(forward(x_small, params_minus)[0], y_small)
    numerical = (loss_plus - loss_minus) / (2 * epsilon)
    probs, cache = forward(x_small, params)
    analytic = backward(y_small, params, cache)['d' + param_name][index]
    return numerical, analytic, abs(numerical - analytic)
# Actualizacion y metricas
def update_parameters(params, grads, lerning_rate):
    for name in ["w1", "b1", "w2", "b2"]:
        params[name] -= lerning_rate * grads["d" + name]
    return params
def predict(x, params):
    probs, _ = forward(x, params)
    return probs.argmax(axis=1), probs
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)
def confusion_matrix_np(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm

# Entrenamiento completo
def train_network(X_train, y_train, X_val, y_val, n_hidden=8, lr=0,epochs=600, seed=42):
    params = initialize_parameters(X_train.shape[1], n_hidden, len(np.unique(y_train)), seed)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        probs, cache = forward(X_train, params)
        loss = cross_entropy_loss(probs, y_train)
        grads = backward(y_train, params, cache)
        params = update_parameters(params, grads, lr)
        y_pred_train, train_probs = predict(X_train, params)
        y_pred_val, val_probs = predict(X_val, params)
        val_loss = cross_entropy_loss(val_probs, y_val)

        history["train_loss"].append(loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(accuracy_score(y_train, y_pred_train))
        history["val_acc"].append(accuracy_score(y_val, y_pred_val))
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: train_loss = {loss:.4f}, val_loss = {val_loss:.4f}")   
    return params, history
# Reporte de errores
def error_report(ids_subset, y_true, y_pred, probs, index_to_class):
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
    x = np.array(features, np.float32).reshape(1, -1)
    x_normalized = (x - mean) / std
    probs, _ = forward(x_normalized, params)
    pred_idx = int(probs.argmax(axis=1)[0])
    return index_to_class[pred_idx], probs[0]

# Graficas
def plot_history(history):
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")
    plt.title("Evolution of loss during training")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()

    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Evolution of accuracy during training")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=150)
    plt.show()

# Programa principal
def main():
    print("---- Practiva 3: Red neuronal desde cero ----")
    X, y, ids = generate_argumented_dataset(base_samples)
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    print("Distribution: ", {index_to_class[k]: int(np.sum(y==k)) for k in np.unique(y)})
    #particion
    train_idx, validation_idx, test_idx = split_indices_stratified(y, seed=42)
    X_train, y_train = X[train_idx], y[train_idx]
    X_validation, y_validation = X[validation_idx], y[validation_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    #normalizacion sin fuga de datos
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    X_train_n = (X_train - mean) / std
    X_validation_n = (X_validation - mean) / std
    X_test_n = (X_test - mean) / std

    #pruebas rapidas de activacion
    Z_demo = np.array([[-2.0,0.0,3.0]])
    print("reLU", relu(Z_demo))
    print("reLU derivative: ", relu_derivative(Z_demo))
    logits_demo = np.array([[2.0,1.0,0.1]])
    print("softmax: ", softmax(logits_demo))
    probs_demo = softmax(logits_demo)
    print("softmax demo: ", np.round(probs_demo,4) )
    print("sum softmax: ", probs_demo.sum(axis=1) )
    
    #verificacion de shapes e inicializacion
    params_demo = initialize_parameters(n_features=3, n_hidden=4, n_classes=2, seed=42)
    for name, value in params_demo.items():
        print(f"{name}: {value.shape}")
    #gradient checking
    num, ana, diff = numerical_gradient_check(X_train_n[:5], y_train[:5], params_demo)
    print(f"Gradient check[0,0] -> numeric = {num:.8f}, analitic={ana:.8f}, diference={diff:.8f}")
    
    #entrenamiento
    params_trained, history = train_network(
        X_train_n, y_train, X_validation_n, y_validation,n_hidden=8,lr=0.03,epochs=600, seed=42
    )
    #evaluacion de train, validation y test
    y_pred_train = predict(X_train_n, params_trained)
    y_pred_val = predict(X_validation_n, params_trained)
    y_pred_test, probs_test = predict(X_test_n, params_trained)

    print("Final metrics------")
    print("Train accuracy: ", accuracy_score(y_train, y_pred_train))
    print("Validation accuracy: ", accuracy_score(y_validation, y_pred_val))
    print("Test accuracy: ", accuracy_score(y_test, y_pred_test))

    cm = confusion_matrix_np(y_test, y_pred_test, len(class_to_index))
    print("Matriz de confusion test")
    print("Rows = real class; columns = predicted class")
    print(cm)

    print("First predictions (test)")
    for i in range(min(10, len(y_test))):
      print(
          ids[test_idx[i]],
          "real=",index_to_class[y_test[test_idx[i]]],
          "predicted=",index_to_class[y_pred_test[i]],
          "probs=", np.round(probs_test[i],3)
      )
    errors = error_report(ids[test_idx], y_test, y_pred_test, probs_test, index_to_class)
    print("Errores detectados:", len(errors))
    for item in errors[:5]:
      print(item)



    

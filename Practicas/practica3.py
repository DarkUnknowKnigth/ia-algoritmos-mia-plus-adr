#inferencia: consiste en usar parametros ya existentes para transformar una entrada en una salida
#Entrenamiento: Modificar esos parametros con base en una senal de error
import numpy as np
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
    b1 = np.zeros(1, n_hidden)
    w2 = range.normal(0, np.sqrt(2/n_hidden), size=(n_hidden, n_classes))
    b2 = np.zeros(1, n_classes)
    return {"w1":w1, "b1":b1, "w2":w2, "b2":b2}
params = initialize_parameters( n_features=4, n_hidden=10, n_classes=3, seed=42)


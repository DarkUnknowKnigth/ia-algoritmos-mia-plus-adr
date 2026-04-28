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
print("Dimension de x", X.shape)
print("Dimension de y", y.shape)
print("Distribucion: " ,{index_to_class[k]: int(np.sum(y==k)) for k in np.unique(y)} )
print("Dimension de ids", len(ids ))
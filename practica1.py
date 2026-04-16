import numpy as np
#crear un dataset sintetico
samples = [
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
#conteo de elementos
print("total de elementos", len(samples),"\n\n===============+++++===============\n")
#verificar la estructura (1) de la lista
print("primera muestra",samples[0],"\n\n===============+++++===============\n")

#extraer etiquetas y clases
## esto es como un destructuracion y map en python 
labels = [ sample["label"] for sample in samples ]
print("etiquetas", labels,"\n\n===============+++++===============\n")
print("etiquetas totales", len(labels),"\n\n===============+++++===============\n")
#clases unicas (cuantos conjuntos de etiquetas existen)
unique_label = set (labels)
print("clases unicas",unique_label,"\n\n===============+++++===============\n")
print("clases totales",len(unique_label),"\n\n===============+++++===============\n")
#ordenar y construir el codificador de etiquetas (traductores de numeros a texto y viceversa para su uso en calculos)
sorted_labels = sorted(unique_label)
print("etiquetas ordenadas",sorted_labels,"\n\n===============+++++===============\n")
class_to_index = {label:idx for idx,label in enumerate(sorted_labels)} 
print("codificador de etiquetas",class_to_index,"\n\n===============+++++===============\n")
index_to_class = {idx:label for label,idx in class_to_index.items()} 
print("decodificador de etiquetas",index_to_class,"\n\n===============+++++===============\n")

#codificar las etiquetas del dataset - y (pasamos las etiquetas de texto a numeros y creamos un vector con la traduccion)
y = np.array([class_to_index[sample["label"]] for sample in samples], dtype= np.int64)
print("etiquetas codificadas y",y,"\n\n===============+++++===============\n")
#construir la matriz de caracteristicas - x
x = np.array([ sample["features"] for sample in samples], dtype=np.float32)
print("matriz de caracteristicas X",x,"\n\n===============+++++===============\n")
print("Dimensiones de matriz de caracteristicas x",x.shape,"\n\n===============+++++===============\n")
# extraccion de id y metadatos 
ids = [sample["id"] for sample in samples]
metadata_list = [sample["metadata"] for sample in samples]
print("ids ",ids,"\n\n===============+++++===============\n")
print("metadatos ",metadata_list,"\n\n===============+++++===============\n")
print("primera muestra de metadatos", metadata_list[0],"\n\n===============+++++===============\n")
#cortar arreglos
porcion = 2
print("pequeña porcion ",[item[0] for item in np.array_split(metadata_list, porcion)],"\n\n===============+++++===============\n")
for item in metadata_list[2:5]:
    print(item,"\n\n===============+++++===============\n")
#verificar si los ids son unicos
def validar_unique_ids(samples):
    ids = [sample["id"] for sample in samples]
    unique_ids = set(ids)
    return len(ids) == len(unique_ids)

##verificacion de ids
print("Todos los ids son unicos??", validar_unique_ids(samples),"\n\n===============+++++===============\n")

#deteccion de ids duplicados
def encontrar_duplicados(samples):
    vistos = set()
    duplicados = set()
    for sample in samples:
        sample_id = sample["id"]
        #marcamos los visto y en la siguiente iteracion buscamos si se duplica
        if sample_id in vistos:
            duplicados.add(sample_id)
        else:
            vistos.add(sample_id)
    return duplicados

duplicados = encontrar_duplicados(samples)
print("ids duplicados",duplicados,"\n\n===============+++++===============\n")

#validar longitud de features
def valida_longitud_de_features(samples):
    #sacamos las longitudes y poner en array
    longitudes = [ len(sample["features"]) for sample in samples]
    # validar y retornar tupla [validacion, longitudes]
    return len(set(longitudes)) == 1, longitudes

longitud_valida, longitudes = valida_longitud_de_features(samples)
print("Todas las muestras tienen la misma longitud de feature??", longitud_valida,"\n\n===============+++++===============\n")
print("longitudes de las muestras", longitudes,"\n\n===============+++++===============\n")

#validar tipos de datos numericos
def validar_tipos_de_datos(samples):
    for sample in samples:
        for value in sample["features"]:
            #se agrega un try por si es vacio y no puede hacer esta verificacion
            try:
                if not isinstance(value, (int, float)):
                    return False
            except Exception as e:
                return False
    return True

tipos_validos = validar_tipos_de_datos(samples)
print("Los valores de features son numericos??", tipos_validos,"\n\n===============+++++===============\n")

#resumen estadistico basico del dataset
#contar muestra por clase
def class_distribution(samples):
    distribution = {}
    for sample in samples:
        label = sample['label']
        distribution[label] = distribution.get(label,0) + 1
    return distribution
dist = class_distribution(samples)
print("Distribucion por clases ", dist,"\n\n===============+++++===============\n")
#resumen estadistico basico del dataset
#resumen estadistico de x
#importante para verificar si esta balanceada la informacion 
def summarize_features(x):
    #calculo de las media, desviacion , minimo y maximo
    sumary = {
        "mean":np.mean(x, axis=0),
        "std":np.std(x, axis=0),
        "min":np.min(x, axis=0),
        "max":np.max(x, axis=0)
    }
    return sumary
summary = summarize_features(x)
print("Resumen estadistico por caracteristicas: ", summary,"\n\n===============+++++===============\n")
for key,value in summary.items():
    print(f"{key}: {value}","\n\n===============+++++===============\n")

#busquedas y consultas del data set
#buscar por id
def get_sample_by_id(samples, target_id):
    for sample in samples:
        if sample["id"] == target_id:
            return sample
    return None
sample_noexiste = get_sample_by_id(samples,"sample_000")
print("muestra no existe",sample_noexiste,"\n\n===============+++++===============\n")
sample = get_sample_by_id(samples,"sample_001")
print("muestra buscada: ",sample,"\n\n===============+++++===============\n")
#estratificacon (segmentacion de conjuntos) 
# Filtrar muestras por etiqueta (clase)

def filter_by_label(samples, target_label):
    return [sample for sample in samples if sample["label"] == target_label]

warning_samples = filter_by_label(samples,'warning')
print("muestras de clase warning", warning_samples,"\n\n===============+++++===============\n")
for item in warning_samples:
    print(item,"\n\n===============+++++===============\n")

#busquedas 
#buscar por fuente
def filter_by_source(samples, target_source):
    return [sample for sample in samples if sample["metadata"]["source"] == target_source]
samples_sensor_b = filter_by_source(samples,"sensor_b")
print("muestras de sensor b", samples_sensor_b,"\n\n===============+++++===============\n")
for item in samples_sensor_b:
    print(item,"\n\n===============+++++===============\n")

#consolidacion del dataset
dataset = {
    "x":x,
    "y":y,
    "ids":ids,
    "samples":samples,
    "class_to_index":class_to_index,
    "index_to_class":index_to_class,
    "metadata": metadata_list
}
print("claves del dataset: ", dataset.keys(),"\n\n===============+++++===============\n")
# print("valores del dataset: ", dataset.items(),"\n\n===============+++++===============\n")
#particion del dataset (Entrenamiento, validacion, prueba)
#separamos los datos para su uso futuro delimitando cuanto servira para entrenar, validar y probar
def split_dataset(x,y,ids, train_ratio = 0.6, validation_ratio =0.2, test_ratio = 0.2):
    n=len(x)
    indices = np.arange(n)
    #en que valor rermina el entremamineot valores totales por el porcentaje el entrenamiento
    train_end = int(n*train_ratio)
    #tomar el valor anteriro ocupado mas el valor de validacion
    validation_end = train_end + int(n*validation_ratio)
    #test valor 
    test_end = +validation_end + int(n*test_ratio)
    #hacer un array slice de los valores por las entradas de rango
    train_idx = indices[:train_end]
    validation_idx = indices[train_end:validation_end]
    test_idx = indices[validation_end:]
    # separando datos
    splits = {
        "train":{
            "x":x[train_idx],
            "y":y[train_idx],
            "ids":[ids[i] for i in train_idx]
        },
        "validation":{
            "x":x[validation_idx],
            "y":y[validation_idx],
            "ids":[ids[i] for i in validation_idx]
        }, "test":{
            "x":x[test_idx],
            "y":y[test_idx],
            "ids":[ids[i] for i in test_idx]
        },
    }
    return splits
splits = split_dataset(x,y,ids)
print("Train IDs: ",splits['train']['ids'],"\n\n===============+++++===============\n")
print("Validation IDs: ",splits['validation']['ids'],"\n\n===============+++++===============\n")
print("Test IDs: ",splits['test']['ids'],"\n\n===============+++++===============\n")

#particion del dataset alearoeio (Entrenamiento, validacion, prueba)
#esto es importante para garantizar tomar datos que no tengan un orden o agrupacion
def split_dataset_aleatoreo(x,y,ids, train_ratio = 0.6, validation_ratio =0.2, test_ratio = 0.2,seed = 42):
    np.random.seed(seed)
    n=len(x) 
    indices = np.arange(n)
    np.random.shuffle(indices)
    #en que valor rermina el entremamineot valores totales por el porcentaje el entrenamiento
    train_end = int(n*train_ratio)
    #tomar el valor anteriro ocupado mas el valor de validacion
    validation_end = train_end + int(n*validation_ratio)
    #test valor 
    test_end = +validation_end + int(n*test_ratio)
    #hacer un array slice de los valores por las entradas de rango
    train_idx = indices[:train_end]
    validation_idx = indices[train_end:validation_end]
    test_idx = indices[validation_end:]
    # separando datos
    splits = {
        "train":{
            "x":x[train_idx],
            "y":y[train_idx],
            "ids":[ids[i] for i in train_idx]
        },
        "validation":{
            "x":x[validation_idx],
            "y":y[validation_idx],
            "ids":[ids[i] for i in validation_idx]
        }, "test":{
            "x":x[test_idx],
            "y":y[test_idx],
            "ids":[ids[i] for i in test_idx]
        },
    }
    return splits
splits = split_dataset_aleatoreo(x,y,ids)
print("aleatoreo Train IDs: ",splits['train']['ids'],"\n\n===============+++++===============\n")
print("aleatoreo Validation IDs: ",splits['validation']['ids'],"\n\n===============+++++===============\n")
print("aleatoreo Test IDs: ",splits['test']['ids'],"\n\n===============+++++===============\n")
#actividad1_minombre.zip

def build_id_index(samples):
    return {sample["id"]: sample for sample in samples}
# construir catalogo de indexes
id_index = build_id_index(samples)
print("consulta rapida por id ","\n\n===============+++++===============\n")
print(id_index["sample_001"],"\n\n===============+++++===============\n")

def find_duplicate_features(samples):
    seen = {}
    duplicados = [] 
    for sample in samples:
        feature_tuple = tuple(sample["features"])
        if feature_tuple in seen:
            duplicados.append(sample["id"])
        else:
            seen[feature_tuple] = sample["id"]
    return duplicados

feature_duplicates = find_duplicate_features(samples)
print("Elementos duplicados por features:", feature_duplicates,"\n\n===============+++++===============\n")
# agregar muestra y validar caracteristicas para evitar errores
def add_sample(samples, new_sample, expected_feature_length):
    required_keys = {"id", "features", "label", "metadata"}
    if set(new_sample.keys()) != required_keys:
        raise ValueError("La muestra no contiene exactamente las claves requeridas")
    if len(new_sample["features"]) != expected_feature_length:
        raise ValueError("Longitud de caracteristicas no coincide con las especificaciones:", expected_feature_length )
    existing_ids = {sample["id"] for sample in samples}
    if new_sample["id"] in existing_ids:
        raise ValueError("El id ya existe en el dataset")
    for value in new_sample["features"]:
        if not isinstance(value, (int, float)):
            raise ValueError("Las caracteristicas deben de ser numeros")
    samples.append(new_sample)
#muestra de prueba
new_sample = {
    "id": "sample_011",
    "features": [0.1, 0.2, 0.3, 0.4],
    "label": "normal",
    "metadata": {
        "source": "sensor_a",
        "quality": "high"
    }
}
try:
    print('Muestras acutales: ', len(samples),"\n\n===============+++++===============\n")
    add_sample(samples, new_sample,4)
except Exception as e:
    print(e,"\n\n===============+++++===============\n")
print('Muestras despues de insertar: ', len(samples),"\n\n===============+++++===============\n")
print(samples[len(samples) -1],"\n\n===============+++++===============\n")
#creando una funcion para reacondicionar los datos del dataset cuando es modificado
def rebuild_dataset(samples):
    labels = [sample["label"] for sample in samples]
    unique_labels = set(labels)
    class_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_class = {idx: label for label, idx in class_to_index.items()}
    x = np.array([sample["features"] for sample in samples], dtype=np.float32)
    y = np.array([class_to_index[sample["label"]] for sample in samples], dtype=np.int64)
    ids = [sample["id"] for sample in samples]
    metadata = [sample["metadata"] for sample in samples]
    dataset = {
        "x": x,
        "y": y,
        "ids": ids,
        "samples": samples,
        "class_to_index": class_to_index,
        "index_to_class": index_to_class,
        "metadata": metadata
    }
    return dataset
dataset = rebuild_dataset(samples)
print("nueva dimsension del dataset X: ", dataset["x"].shape,"\n\n===============+++++===============\n")
print("nueva dimsension del dataset Y: ", dataset["y"],"\n\n===============+++++===============\n")
print(dataset,"\n\n===============+++++===============\n")
#finalizacion de practica 1

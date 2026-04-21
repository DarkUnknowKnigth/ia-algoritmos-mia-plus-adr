import numpy as np

class Builder:
    """
    Clase encargada de la construcción y partición de datasets procesados.
    """
    samples = []
    dataset = {}
    def __init__(self,samples):
        """
        Inicializar los datos.
        """
        self.samples = samples
    def evaluate_quality(self):
        """
        Validar si los features son numericos y estan presentes
        """
        validated_samples = []
        for sample in self.samples:
            flag_valid = 2
            #asegurandonos que las clases sean texto valido
            if sample["label"] == None or sample["label"] == '':
                flag_valid -=1
            #asegurandonos que los ids existan y no esten vacios
            if sample["id"] == None or sample["id"] == '':
                flag_valid -=1
            for feature in sample["features"]:
                if not isinstance(feature, (int, float)):
                    flag_valid -=1
                #valido que las features tengan algo que sea validao como numeros mayores a cero y no cadenas vacias
                if feature != None and feature != '' and float(feature) < 0:
                    flag_valid -=1
            if flag_valid == 2:
                validated_samples.append(sample)
        self.samples = validated_samples     
    def avoid_duplicity(self):
        """
        Elimina muestras duplicadas basándose en su ID y en la combinación de sus características.
        """
        seen_ids = set()
        seen_features = set()
        unique_samples = []

        for sample in self.samples:
            sample_id = sample["id"]
            feature_tuple = tuple(sample["features"])

            if sample_id not in seen_ids and feature_tuple not in seen_features:
                seen_ids.add(sample_id)
                seen_features.add(feature_tuple)
                unique_samples.append(sample)
        
        self.samples = unique_samples
        
    def build_dataset(self):
        """
            Construir todo lo necesario para que el dataset siga lo visto en clase
        """
        labels = [sample["label"] for sample in self.samples]
        unique_labels = sorted(set(labels))
        class_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        index_to_class = {idx: label for label, idx in class_to_index.items()}
        x = np.array([sample["features"] for sample in self.samples], dtype=np.float32)
        y = np.array([class_to_index[sample["label"]] for sample in self.samples], dtype=np.int64)
        ids = [sample["id"] for sample in self.samples]
        metadata = [sample["metadata"] for sample in self.samples]
        
        self.dataset = {
            "x": x,
            "y": y,
            "ids": ids,
            "samples": self.samples,
            "class_to_index": class_to_index,
            "index_to_class": index_to_class,
            "metadata": metadata
        }
        return self.dataset
    def split_dataset(self, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2, seed=None):
        """
        Divide el dataset en conjuntos de entrenamiento, validación y prueba.
        """
        import numpy as np
        if seed is not None:
            np.random.seed(seed)
            
        x = self.dataset["x"]
        y = self.dataset["y"]
        ids = self.dataset["ids"]
        n = len(x)
        indices = np.arange(n)
        
        if seed is not None:
            np.random.shuffle(indices)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * validation_ratio)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        return {
            "train": {
                "x": x[train_idx],
                "y": y[train_idx],
                "ids": [ids[i] for i in train_idx]
            },
            "validation": {
                "x": x[val_idx],
                "y": y[val_idx],
                "ids": [ids[i] for i in val_idx]
            },
            "test": {
                "x": x[test_idx],
                "y": y[test_idx],
                "ids": [ids[i] for i in test_idx]
            }
        }
    def normalize_labels(self):
        translator = {'warn':'warning','critico':'critical'}
        tuned_data = []
        for sample in self.samples:
            if sample['label'] in translator:
                normalized_label = translator.get(sample['label'])
                sample['label'] = normalized_label
                tuned_data.append(sample)
            else:
                tuned_data.append(sample)
        self.samples = tuned_data
    def normalize_metadata(self, key):
        translator = {}
        tuned_data = []
        if key == 'calidad_medicion':
            translator = {'alta':'high','media':'medium','baja':'low'}
        for sample in self.samples:
            if sample['metadata'][key] in translator:
                normalized_metadata = translator.get(sample['metadata'][key])
                sample['metadata'][key] = normalized_metadata
                tuned_data.append(sample)
            else:
                tuned_data.append(sample)
        self.samples = tuned_data
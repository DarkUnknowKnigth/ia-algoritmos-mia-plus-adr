import numpy as np

class Builder:
    """
    Clase encargada de la construcción y partición de datasets procesados.
    """
    samples = []
    dataset = {}
    def __init__(self,samples):
        """
        Reconstruye la estructura del dataset a partir de una lista de muestras.
        """
        self.samples = samples
    def evaluate_quality(self):
        """
        Validar si los features son numericos y estan presentes
        """
        validated_samples = []
        for sample in self.samples:
            flag_valid = 2
            for feature in sample["features"]:
                if not isinstance(feature, (int, float)):
                    flag_valid -=1
            if flag_valid == 2:
                validated_samples.append(sample)
        self.samples = validated_samples
    def build_dataset(self):
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
            "val": {
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
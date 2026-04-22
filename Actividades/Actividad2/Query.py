import numpy as np
from Builder import DatasetInterface

class Query:
    """
    Clase para realizar búsquedas y consultas sobre el dataset procesado.
    """
    def __init__(self, dataset: DatasetInterface):
        self.dataset = dataset
        self.id_index = {sample["id"]: sample for sample in dataset["samples"]}

    def get_by_id(self, target_id):
        """Busca una muestra específica por su ID."""
        return self.id_index.get(target_id)

    def filter_by_label(self, target_label):
        """Filtra las muestras que pertenecen a una clase específica."""
        return [s for s in self.dataset["samples"] if s["label"] == target_label]

    def get_stats(self):
        """Calcula estadísticas básicas de las características (X)."""
        x = self.dataset["x"]
        return {
            "mean": np.mean(x, axis=0),
            "std": np.std(x, axis=0),
            "min": np.min(x, axis=0),
            "max": np.max(x, axis=0),
            "count": len(x)
        }

    def class_distribution(self):
        """Devuelve el conteo de muestras por cada etiqueta."""
        dist = {}
        for sample in self.dataset["samples"]:
            label = sample["label"]
            dist[label] = dist.get(label, 0) + 1
        return dist
    
    def filter_by_metadata(self, key, value):
        """Filtra las muestras por metadata."""
        return [s for s in self.dataset["samples"] if s["metadata"].get(key) == value]
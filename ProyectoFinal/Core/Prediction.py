import numpy as np

from Core.Builder import DatasetInterface
from Core.Layer import *



class Prediction:
    """
    Clase para realizar inferencia utilizando una red neuronal de dos capas.
    """
    dataset: DatasetInterface | None = None
    class_names = {}
    X , Y = np.array([], np.float32),np.array([], np.float32)
    layers: list[Layer] = []
    layers_map: dict[int, Layer] = {}
    predictions = []


    prediction = np.array([], np.float32)
    def __init__(self, dataset: DatasetInterface):
        self.dataset =dataset
        self.class_names = dataset["index_to_class"]
        self.X = dataset["x"]
        self.Y = dataset["y"]
    def addLayer(self, layer: Layer):
        """
          Agrega una capa a la matris de capas
        """
        self.layers.append(layer)
        self.layers_map = { idx: layer for idx, layer in enumerate(self.layers) }

    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    @staticmethod
    def softmax(x):
        x_shifted = x - np.max(x)
        exp_values = np.exp(x_shifted)
        return exp_values / np.sum(exp_values)
    @staticmethod
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def accuracy_score(self, y_real, y_predicted):
        return np.mean(y_real == y_predicted)
    def predict(self, layer_index1, layer_index2):
        """
        Realiza el forward pass para todas las muestras en self.X.
        """
        layer1 = self.layers_map[layer_index1]
        layer2 = self.layers_map[layer_index2]
        self.predictions = [self.predict_single(x, layer1, layer2, idx) for idx, x in enumerate(self.X)]
        return self.predictions
    
    def predict_single(self, x, layer1: Layer, layer2: Layer, idx=None):
        # Capa 1 (Oculta)
        a1 = layer1.neuron_forward(x, self.relu)
        # Capa 2 (Salida)
        z2 = layer2.neuron_forward(a1)
        probs = self.softmax(z2)
        
        y_pred = np.argmax(probs)
        
        # Calcular accuracy score correcto para esta muestra
        if idx is not None and self.Y is not None and len(self.Y) > idx:
            acc = float(self.Y[idx] == y_pred)
        else:
            if self.Y is not None and np.isscalar(self.Y):
                acc = float(self.Y == y_pred)
            else:
                acc = 0.0
        
        return {
            "hidden": a1,
            "scores": z2,
            "probabilities": probs,
            "class_index": y_pred,
            "class_name": self.class_names.get(y_pred, "unknown"),
            "accuracy_score": acc
        }


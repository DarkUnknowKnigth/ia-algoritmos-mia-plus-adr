import numpy as np

class Layer:
    """
    Representa una capa densa dentro de la red neuronal.
    """
    def __init__(self, weights: np.ndarray, bias: np.ndarray):
        self.weights = weights
        self.bias = bias
    def neuron_forward(self, x, activation=None):
        """
        Realiza el forward pass para una sola muestra.
        """
        self.validate_dense_shapes(x)
        z = np.dot(x, self.weights) + self.bias
        if activation:
            return activation(z)
        return z
    def validate_dense_shapes(self, X):
        if X.ndim != 1:
            raise ValueError(f"X debe ser una matriz bidimensional")
        if self.weights.ndim != 2:
            raise ValueError(f"W debe ser una matriz bidimensional {self.weights.shape[0]}x{self.weights.shape[1]}")
        if self.bias.ndim != 1:
            raise ValueError("b debe ser un vector unidimensional")
        if X.shape[0] != self.weights.shape[0]:
            raise ValueError(f"El numero de columnas de X {X.shape[0]} debe ser igual al numero de filas de W {self.weights.shape[0]}")
        if self.weights.shape[1] != self.bias.shape[0]:
            raise ValueError("El numero de columnas de W debe ser igual al numero de filas de b")   
import numpy as np

class Layer:
    """
    Representa una capa densa dentro de la red neuronal.
    Soporta propagación para entradas de cualquier número de dimensiones (1D, 2D/batch, N-D).
    """
    def __init__(self, weights: np.ndarray, bias: np.ndarray):
        self.weights = weights
        self.bias = bias

    def forward(self, x: np.ndarray, activation=None) -> np.ndarray:
        """
        Realiza el forward pass para una muestra, lote (batch) o tensor multidimensional x.
        """
        self.validate_dense_shapes(x)
        # z = x @ W + b. En NumPy, np.matmul o @ maneja múltiples dimensiones correctamente
        # aplicando la multiplicación en la última dimensión de x.
        z = np.matmul(x, self.weights) + self.bias
        if activation:
            return activation(z)
        return z

    def neuron_forward(self, x: np.ndarray, activation=None) -> np.ndarray:
        """
        Conserva compatibilidad con la firma original del proyecto.
        """
        return self.forward(x, activation)

    def validate_dense_shapes(self, X: np.ndarray):
        """
        Valida que las dimensiones de entrada, pesos y sesgos sean compatibles.
        """
        if self.weights.ndim != 2:
            raise ValueError(f"Los pesos W deben ser una matriz bidimensional (input_dim x output_dim), forma actual: {self.weights.shape}")
        
        # Validamos que X tenga al menos 1 dimensión
        if X.ndim == 0:
            raise ValueError("La entrada X debe ser al menos unidimensional.")
            
        # El último eje de X debe coincidir con el primer eje de W (input_dim)
        input_dim = X.shape[-1]
        if input_dim != self.weights.shape[0]:
            raise ValueError(
                f"La dimensión de características de entrada ({input_dim}) no coincide "
                f"con las filas de W ({self.weights.shape[0]})."
            )

        # El sesgo b puede ser 1D (output_dim,) o 2D (1, output_dim), etc.
        # Validamos que el último eje de b coincida con el segundo eje de W (output_dim)
        if self.bias.ndim == 0:
            raise ValueError("El sesgo b debe ser al menos unidimensional.")
            
        output_dim = self.weights.shape[1]
        if self.bias.shape[-1] != output_dim:
            raise ValueError(
                f"La dimensión del sesgo ({self.bias.shape[-1]}) no coincide "
                f"con las columnas de W ({output_dim})."
            )
   
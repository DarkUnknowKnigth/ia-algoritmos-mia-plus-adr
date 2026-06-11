import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from Core.Layer import Layer

class Trainer:
    """
    Clase responsable de entrenar la red neuronal, realizar la retropropagación,
    actualizar parámetros, calcular métricas (pérdida de entropía cruzada, precisión),
    realizar validación numérica de gradientes, ejecutar experimentos y graficar el historial.
    Diseñado bajo principios SOLID y DRY para ser modular y reutilizable.
    """
    def __init__(self, train_data: Dict[str, Any], val_data: Dict[str, Any] = None):
        """
        Inicializa el entrenador con conjuntos de datos.
        train_data y val_data son diccionarios que deben contener al menos 'x' e 'y'.
        """
        self.train_data = train_data
        self.val_data = val_data
        self.layers: List[Layer] = []
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        self.mean = None
        self.std = None

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(np.float32)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        # Estabilidad numérica restando el máximo por fila (soporta batches 2D y N-D)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    @staticmethod
    def cross_entropy_loss(probs: np.ndarray, y_true: np.ndarray) -> float:
        n = len(y_true)
        # Selecciona las probabilidades correspondientes a las clases verdaderas.
        # Funciona asumiendo y_true plano (1D).
        selected = probs[np.arange(n), y_true]
        return -float(np.mean(np.log(selected + 1e-9)))

    def initialize_network(self, n_features: int, n_hidden: int, n_classes: int, seed: int = 42) -> List[Layer]:
        """
        Inicializa las capas usando He normal initialization.
        """
        rng = np.random.default_rng(seed)
        w1 = rng.normal(0, np.sqrt(2 / n_features), size=(n_features, n_hidden))
        b1 = np.zeros((1, n_hidden))
        w2 = rng.normal(0, np.sqrt(2 / n_hidden), size=(n_hidden, n_classes))
        b2 = np.zeros((1, n_classes))
        
        self.layers = [
            Layer(w1, b1),
            Layer(w2, b2)
        ]
        return self.layers

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Pase hacia adelante (forward pass) compatible con lotes de cualquier dimensión (N-D).
        """
        if len(self.layers) < 2:
            raise ValueError("La red neuronal no ha sido inicializada.")
            
        Z1 = self.layers[0].forward(X)
        A1 = self.relu(Z1)
        logits = self.layers[1].forward(A1)
        probs = self.softmax(logits)
        
        cache = {
            'X': X,
            'Z1': Z1,
            'A1': A1,
            'logits': logits,
            'probs': probs
        }
        return probs, cache

    def backward(self, y_true: np.ndarray, cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Realiza el pase hacia atrás (backpropagation) para calcular gradientes.
        """
        X = cache['X']
        A1 = cache['A1']
        Z1 = cache['Z1']
        probs = cache['probs']
        w2 = self.layers[1].weights
        n = X.shape[0]

        # Gradientes de salida
        dlogits = probs.copy()
        dlogits[np.arange(n), y_true] -= 1.0
        dlogits /= n
        
        dw2 = A1.T @ dlogits
        db2 = np.sum(dlogits, axis=0, keepdims=True)

        # Propagación del error a la capa oculta
        da1 = dlogits @ w2.T
        dz1 = da1 * self.relu_derivative(Z1)
        
        dw1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {
            'dw1': dw1,
            'db1': db1,
            'dw2': dw2,
            'db2': db2
        }

    def update_parameters(self, grads: Dict[str, np.ndarray], learning_rate: float):
        """
        Actualiza los pesos de las capas correspondientes con los gradientes.
        """
        self.layers[0].weights -= learning_rate * grads["dw1"]
        self.layers[0].bias -= learning_rate * grads["db1"]
        self.layers[1].weights -= learning_rate * grads["dw2"]
        self.layers[1].bias -= learning_rate * grads["db2"]

    def numerical_gradient_check(self, X_small: np.ndarray, y_small: np.ndarray, epsilon: float = 1e-7) -> Tuple[float, float, float]:
        """
        Verifica numéricamente los gradientes analíticos sobre el peso w1[0,0].
        """
        if len(self.layers) < 2:
            raise ValueError("La red neuronal no ha sido inicializada.")
            
        w1_orig = self.layers[0].weights.copy()
        
        # Perturbación hacia adelante (w1[0,0] + epsilon)
        self.layers[0].weights[0, 0] = w1_orig[0, 0] + epsilon
        probs_plus, _ = self.forward(X_small)
        loss_plus = self.cross_entropy_loss(probs_plus, y_small)
        
        # Perturbación hacia atrás (w1[0,0] - epsilon)
        self.layers[0].weights[0, 0] = w1_orig[0, 0] - epsilon
        probs_minus, _ = self.forward(X_small)
        loss_minus = self.cross_entropy_loss(probs_minus, y_small)
        
        # Restaurar el peso original
        self.layers[0].weights[0, 0] = w1_orig[0, 0]
        
        # Gradiente numérico aproximado
        numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Gradiente analítico
        probs, cache = self.forward(X_small)
        grads = self.backward(y_small, cache)
        analytic_grad = grads['dw1'][0, 0]
        
        difference = abs(numerical_grad - analytic_grad)
        return numerical_grad, analytic_grad, difference

    def train(self, n_hidden: int = 8, lr: float = 0.03, epochs: int = 600, seed: int = 42) -> Tuple[List[Layer], Dict[str, List[float]]]:
        """
        Entrena el modelo y registra el historial de pérdidas y precisión.
        """
        X_train = self.train_data["x"]
        y_train = self.train_data["y"]
        
        # Calcular estadísticas de normalización sobre el conjunto de entrenamiento
        self.mean = X_train.mean(axis=0)
        self.std = X_train.std(axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)
        
        X_train_n = (X_train - self.mean) / self.std
        
        n_features = X_train_n.shape[1]
        if "class_to_index" in self.train_data:
            n_classes = len(self.train_data["class_to_index"])
        else:
            n_classes = int(np.max(y_train) + 1)
        
        self.initialize_network(n_features, n_hidden, n_classes, seed)
        
        X_val_n = None
        if self.val_data is not None:
            X_val_n = (self.val_data["x"] - self.mean) / self.std
            y_val = self.val_data["y"]
            
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        
        for epoch in range(epochs):
            # Pase hacia adelante y cálculo de pérdida
            probs, cache = self.forward(X_train_n)
            loss = self.cross_entropy_loss(probs, y_train)
            
            # Retropropagación y actualización de parámetros
            grads = self.backward(y_train, cache)
            self.update_parameters(grads, lr)
            
            # Evaluar métricas al final de la época
            y_pred_train, train_probs = self.predict(X_train_n)
            train_acc = self.accuracy_score(y_train, y_pred_train)
            
            self.history["train_loss"].append(loss)
            self.history["train_acc"].append(train_acc)
            
            if self.val_data is not None:
                val_probs, _ = self.forward(X_val_n)
                val_loss = self.cross_entropy_loss(val_probs, y_val)
                y_pred_val, _ = self.predict(X_val_n)
                val_acc = self.accuracy_score(y_val, y_pred_val)
                
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
            else:
                self.history["val_loss"].append(0.0)
                self.history["val_acc"].append(0.0)
                
            if epoch % 100 == 0:
                val_str = f", val_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}" if self.val_data is not None else ""
                print(f"Epoch {epoch:03d}: train_loss = {loss:.4f}, train_acc = {train_acc:.4f}{val_str}")
                
        return self.layers, self.history

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza predicciones sobre una matriz X y devuelve clases predichas y probabilidades.
        """
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1), probs

    def accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
        cm = np.zeros((n_classes, n_classes), dtype=np.int32)
        for true, pred in zip(y_true, y_pred):
            cm[true, pred] += 1
        return cm

    def error_report(self, ids_subset: List[str], y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, index_to_class: Dict[int, str]) -> List[Dict[str, Any]]:
        report = []
        for sample_id, true_label, pred_label, prob in zip(ids_subset, y_true, y_pred, probs):
            # Solo agregamos muestras que tengan predicción incorrecta si se busca un reporte de error
            # o listamos todas las muestras. La práctica reporta todas las predicciones del test
            # y señala el conteo total de errores. Cumpliendo con el formato de la práctica:
            if true_label != pred_label:
                report.append({
                    "id": sample_id,
                    "true_label": index_to_class[int(true_label)],
                    "pred_label": index_to_class[int(pred_label)],
                    "acurracy_pred": float(np.max(prob)),
                    "probabilities": np.round(prob, 4).tolist()
                })
        return report

    def save_parameters(self, filename: str = "modelo.npz"):
        if len(self.layers) < 2:
            raise ValueError("No hay parámetros entrenados para guardar.")
        np.savez(
            filename,
            W1=self.layers[0].weights,
            b1=self.layers[0].bias,
            W2=self.layers[1].weights,
            b2=self.layers[1].bias,
            mean=self.mean,
            std=self.std
        )

    def load_parameters(self, filename: str = "modelo.npz") -> Tuple[List[Layer], np.ndarray, np.ndarray]:
        data = np.load(filename)
        w1 = data["W1"]
        b1 = data["b1"]
        w2 = data["W2"]
        b2 = data["b2"]
        self.mean = data["mean"]
        self.std = data["std"]
        
        self.layers = [
            Layer(w1, b1),
            Layer(w2, b2)
        ]
        return self.layers, self.mean, self.std

    def plot_history(self, filename_loss: str = "loss_curve.png", filename_acc: str = "accuracy_curve.png", experiment_name: str = ""):
        """
        Visualiza y guarda los gráficos del historial de entrenamiento.
        """
        epochs = range(len(self.history["train_loss"]))
        title_suffix = f" - {experiment_name}" if experiment_name else ""
        
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history["train_loss"], label="Train Loss", color="royalblue")
        if self.val_data is not None:
            plt.plot(epochs, self.history["val_loss"], label="Val Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Cross Entropy Loss")
        plt.title("Evolution of loss during training" + title_suffix)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(filename_loss, dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history["train_acc"], label="Train Accuracy", color="royalblue")
        if self.val_data is not None:
            plt.plot(epochs, self.history["val_acc"], label="Val Accuracy", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Evolution of accuracy during training" + title_suffix)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(filename_acc, dpi=150)
        plt.close()
        return filename_loss, filename_acc

    def run_experiment(self, experiment: Dict[str, Any], experiment_id: int) -> Tuple[str, Dict[str, Any], List[Layer], Dict[str, List[float]]]:
        """
        Ejecuta un experimento de entrenamiento modular usando los hiperparámetros especificados.
        """
        layers, history = self.train(
            n_hidden=experiment["n_hidden"],
            lr=experiment["lr"],
            epochs=experiment["epochs"],
            seed=experiment["seed"]
        )
        
        X_val = self.val_data["x"]
        y_val = self.val_data["y"]
        X_val_norm = (X_val - self.mean) / self.std
        
        y_pred_val, _ = self.predict(X_val_norm)
        
        if "class_to_index" in self.train_data:
            n_classes = len(self.train_data["class_to_index"])
        else:
            n_classes = int(np.max(y_val) + 1)
            
        metrics = {
            "n_hidden": experiment["n_hidden"],
            "lr": experiment["lr"],
            "val_acc": self.accuracy_score(y_val, y_pred_val),
            "val_loss": history["val_loss"][-1],
            "confusion_matrix": self.confusion_matrix(y_val, y_pred_val, n_classes).tolist()
        }
        
        model_name = f"modelo-{experiment_id:03d}.npz"
        return model_name, metrics, layers, history

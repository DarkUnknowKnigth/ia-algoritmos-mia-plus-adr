from typing import TypedDict, List, Dict, Any, Union
"""
    Interfaz de datos para asegurar la estructura vista en clase de un sample
"""
class SampleInterface(TypedDict):
    id: str
    label: str
    features: List[Union[int, float]]
    metadata: Dict[str, Any]


class Sanitizer:
    """
        Clase que limpia los datos crudos y los lleva a la estructura de buenas practicas de samples como vimos en clase
    """
    data = []
    samples: List[SampleInterface] = []
    trash = []
    duplicated_keys = []
    def __init__(self, data):
        """
        Inicializa la clase con una lista de diccionarios (datos de CSV).
        """
        self.data = data

    def clean_data(self):
        """
        Limpia y normaliza los datos. 
        Elimina espacios en blanco y convierte valores vacíos en None.
        """
        if not self.data:
            return []

        cleaned_data = []
        for row in self.data:
            new_row = {k: (v.strip().lower() if v is not None else None) for k, v in row.items()}
            # Filtrar filas que estén completamente vacías
            if any(new_row.values()):
                cleaned_data.append(new_row)
            else:
                self.trash.append(new_row)
        self.data = cleaned_data
        return self.data

    def remove_duplicates(self, key=None):
        """
        Elimina registros duplicados. Si se proporciona una llave, 
        se basa en esa columna para identificar duplicidad.
        """
        if key:
            seen = set()
            unique_data = []
            for row in self.data:
                val = row[key]
                if val not in seen:
                    seen.add(val)
                    unique_data.append(row)
                else:
                    self.duplicated_keys.append(row)
            self.data = unique_data
        else:
            self.data = [dict(t) for t in {tuple(d.items()) for d in self.data}]
        
        return self.data
    def format_columns(self, column_types):
        """
        Intenta convertir columnas a tipos específicos (int, float, etc.)
        column_types: dict {'nombre_columna': tipo}
        """
        for row in self.data:
            for col, func in column_types.items():
                if col in row and row[col]:
                    try:
                        row[col] = func(row[col])
                    except ValueError:
                        
                        row[col] = None
        return self.data
    def shaper( self, key_id: str,  key_label: str, feature_keys: list[str] = [], metadata_keys: list[str] = []) -> List[SampleInterface]:
        """
        Convertir a una estructura que sigue las buenas practicas {id, label, features, metadata}
        """
        samples = []
        for row in self.data:
            samples.append({
                "id": row[key_id],
                "label": row[key_label],
                "features": [row[key] for key in feature_keys],
                "metadata": {key: row[key] for key in metadata_keys}
            })
        self.samples = samples
        return samples
    def add_sample(self, new_sample, expected_feature_length) -> List[SampleInterface]:
        required_keys = {"id", "label", "features", "metadata"}
        if not required_keys.issubset(new_sample.keys()):
            raise ValueError(f"La muestra debe contener las llaves: {required_keys}")
        if len(new_sample["features"]) != expected_feature_length:
            raise ValueError(f"Longitud de características incorrecta. Se esperaba {expected_feature_length}")
        existing_ids = {s["id"] for s in self.samples if isinstance(s, dict) and "id" in s}
        if new_sample["id"] in existing_ids:
            raise ValueError("El ID ya existe en el conjunto de datos")    
        self.samples.append(new_sample)
        return self.samples
    #componer las cadenas que tienen numero y comas 
    def parse_number(self,x):
        has_comma = ',' in x
        if has_comma:
            try:
                value = float(x.replace(',', '.'))
            except ValueError:
                value = None
        else:
            try:
                value = float(x)
            except ValueError:
                value = None
        return value
    
#crear una clase Sanitizar para tratar un list de datos csv
class Sanitizer:
    data = []
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
                val = row.get(key)
                if val not in seen:
                    seen.add(val)
                    unique_data.append(row)
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
    def shaper( self, key_id: str,  key_label: str, feature_keys: list[str] = [], metadata_keys: list[str] = []):
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
        return samples

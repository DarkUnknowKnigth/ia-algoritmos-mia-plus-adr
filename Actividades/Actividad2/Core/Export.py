import json
import numpy as np
class ExportJSON(json.JSONEncoder):
    """
        Formateo para exportacion de datos en json
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(ExportJSON, self).default(obj)


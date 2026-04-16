# Jose Daniel Morales Ocampo
# MIA + ADR
# Algoritmos de programacion avanzada

#Actividad 2
import csv
#leer un csv y cargarlo en memoria

def load_csv(file_path):
    data = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                data.append(row)
        return data
    except FileNotFoundError:
        print(f"Error: El archivo en {file_path} no fue encontrado.")
        return None
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo: {e}")
        return None

if __name__ == "__main__":
    # Ejemplo de uso
    ruta_archivo = './datos.csv'
    contenido = load_csv(ruta_archivo)
    
    if contenido:
        print(f"Se cargaron {len(contenido)} registros correctamente.")
        
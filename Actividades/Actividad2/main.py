# Jose Daniel Morales Ocampo
# MIA + ADR
# Algoritmos de programacion avanzada

#Actividad 2
import csv
import sys
from Sanitizer import *
from Builder import *

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
    if len(sys.argv) > 1:
        ruta_archivo = sys.argv[1]
    else:
        print("Uso: python main.py <ruta_del_archivo_csv>")
        print("Usando ruta de ejemplo... -> ./datos.csv")
        ruta_archivo = './datos.csv'
        
    # Ejemplo de uso
    contenido = load_csv(ruta_archivo)
    
    if contenido:
        print(f"Se cargaron {len(contenido)} registros correctamente.")
        #sanilizar los datos
        
        sanitizer = Sanitizer(contenido)
        sanitizer.clean_data()
        sanitizer.remove_duplicates()
        sanitizer.format_columns({'ID':int,'Vehiculos_Contados':int,'Velocidad_Promedio':float})
        data = sanitizer.data
        # Exploarcion
        print(data[0])
        samples = sanitizer.shaper('ID','Peligo_colision', ['Vehiculos_Contados','Velocidad_Promedio'], [ 'Fecha', 'Hora','Clima','Estado_Sensor'])
        # Primer elemento del samples para ver el procesamiento
        print(samples[0])
        builder = Builder(samples)
        builder.evaluate_quality()
        dataset = builder.build_dataset()
        for key, value in dataset.items():
            print(f"{key}: {value}")
        trainer = builder.split_dataset()
        for key, value in trainer.items():
            print(f"{key}: {value}")
        
        
        
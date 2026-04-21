# Jose Daniel Morales Ocampo
# MIA + ADR
# Algoritmos de programacion avanzada

#Actividad 2
import csv
import sys
from Sanitizer import *
from Builder import *
from Query import *

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
        print("Usando ruta de ejemplo...")
        ruta_archivo = './dataset_sucio_trafico_urbano_200_muestras.csv'
        
    # Ejemplo de uso
    contenido = load_csv(ruta_archivo)
    
    if contenido:
        print(f"Se cargaron {len(contenido)} registros correctamente.")
        #sanilizar los datos
        id_string = 'sample_id'
        sanitizer = Sanitizer(contenido)
        sanitizer.clean_data()
        sanitizer.remove_duplicates()
        sanitizer.format_columns({id_string:str,'vehiculos_hora':float,'velocidad_promedio_kmh':float,'densidad_vehicular':float,'tiempo_espera_s':float})
        data = sanitizer.data
        # Exploarcion
        print(20*"==","Explorando muestras crudas", 20*"==","\n")
        print(data[0])
        samples = sanitizer.shaper(id_string,'estado', ['vehiculos_hora','velocidad_promedio_kmh','densidad_vehicular','tiempo_espera_s'], [ 'fecha_medicion', 'vehiculos_hora','crucero','zona','calidad_medicion','observacion'])

        # Primer elemento del samples para ver el procesamiento
        print(20*"==","Explorando muestras estructuradas", 20*"==","\n")
        print(samples[0])
        # Crear el dataset
        builder = Builder(samples)
        # Descartar valores con inconsistencias
        builder.evaluate_quality()
        # borrar duplicadidad
        builder.avoid_duplicity()
        # llevar los labels a un estdo similar
        builder.normalize_labels()
        # llevar metadatos que me importan a un estado similar de significado
        builder.normalize_metadata('calidad_medicion')
        # guardar el dataset
        dataset = builder.build_dataset()
        #for key, value in dataset.items():
        #    print(f"{key}: {value}")
        #separar en entrenamiento, validacion y pruebas
        trainer = builder.split_dataset()
        # for key, value in trainer.items():
        #     print(f"{key}: {value}")
        #consultas
        query = Query(dataset)
        print(20*"==","Explorando datasets", 20*"==","\n")
        print(query.get_by_id('tra_099'))
        # filtrar por label
        query_by_label = query.filter_by_label('warning')
        print(20*"==","Comprobacion del filtro metadatos", 20*"==","\n")
        print(set([ s['label'] for s in query_by_label ]))
        # print(query_by_label)
        # filtrar metadatos
        query_by_metadata = query.filter_by_metadata('crucero','cruce_d')
        # print(query_by_metadata)
        print(20*"==","Comprobacion del filtro metadatos", 20*"==","\n")
        print(set([ s['metadata']['crucero'] for s in query_by_metadata ]))

        # estadisticas al dataset
        print(20*"==","Estadisticas básicas", 20*"==","\n")
        stats =query.get_stats()
        for stat in stats.keys():
            print(stat, stats[stat])
        print(20*"==","Explorando distribucion muestral", 20*"==","\n")
        print(query.class_distribution())        

        # estadisticas a subconsultas
        warning_label_dataset = Builder(query_by_label).build_dataset()
        query_warning_dataset = Query(warning_label_dataset)
        print(20*"==","Estadisticas básicas (warning)", 20*"==","\n")
        stats =query_warning_dataset.get_stats()
        for stat in stats.keys():
            print(stat, stats[stat])
        print(20*"==","Explorando distribucion muestral (warning)", 20*"==","\n")
        print(query_warning_dataset.class_distribution())     
        
        # estadisticas a subconsultas
        high_metadata_dataset = Builder(query_by_metadata).build_dataset()
        query_high_dataset = Query(high_metadata_dataset)
        print(20*"==","Estadisticas básicas metadata: calidad(high)", 20*"==","\n")
        stats =query_high_dataset.get_stats()
        for stat in stats.keys():
            print(stat, stats[stat])
        print(20*"==","Explorando distribucion muestral metadata: calidad(high)", 20*"==","\n")
        print(query_high_dataset.class_distribution())     

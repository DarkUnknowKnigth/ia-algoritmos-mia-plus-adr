# Jose Daniel Morales Ocampo
# MIA + ADR
# Algoritmos de programacion avanzada

#Actividad 2
import csv
import sys
import json
import numpy as np
from Core.Export import *
from Core.Sanitizer import *
from Core.Builder import *
from Core.Query import *

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
def export_to_json(dataset, name):
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4, cls=ExportJSON)
    print(20*"==","Dataset exportado como: ",name, 20*"==","\n")
if __name__ == "__main__":
    if len(sys.argv) > 1:
        ruta_archivo = sys.argv[1]
    else:
        print("Uso: python main.py <ruta_del_archivo_csv>")
        print("Usando ruta de ejemplo...")
        ruta_archivo = './Datasets/dataset_sucio_trafico_urbano_200_muestras.csv'
        
    # Ejemplo de uso
    contenido = load_csv(ruta_archivo)
    
    if contenido:
        print(f"Se cargaron {len(contenido)} registros correctamente.")
        #sanilizar los datos
        id_string = 'sample_id'
        sanitizer = Sanitizer(contenido)
        sanitizer.clean_data()
        sanitizer.remove_duplicates()
        sanitizer.format_columns({id_string:str,'vehiculos_hora':sanitizer.parse_number,'velocidad_promedio_kmh':sanitizer.parse_number,'densidad_vehicular':sanitizer.parse_number,'tiempo_espera_s':sanitizer.parse_number})
        data = sanitizer.data
        # Exploarcion
        print(20*"==","Explorando muestras crudas", 20*"==","\n")
        print(data[0])
        samples = sanitizer.shaper(id_string,'estado', ['vehiculos_hora','velocidad_promedio_kmh','densidad_vehicular','tiempo_espera_s'], [ 'fecha_medicion', 'vehiculos_hora','crucero','zona','calidad_medicion','observacion'])

        # Primer elemento del samples para ver el procesamiento
        print(20*"==","Explorando muestras estructuradas", 20*"==","\n")
        first_sample = samples[0]
        print(first_sample)
        features_length = len(first_sample['features'])
        print(20*"==","Agregando muestra estructurada Actual > ",len(samples), 20*"==","\n")

        new_sample = {'id': 'tra_201', 'label': 'critical', 'features': [849.24, 20, 66.87, 178.8], 'metadata': {'fecha_medicion': '2026-04-14', 'vehiculos_hora': 849.24, 'crucero': 'cruce_c', 'zona': 'oriente', 'calidad_medicion': 'alta', 'observacion': 'que calor hace'}}
        samples  = sanitizer.add_sample(new_sample, features_length)
        print(20*"==","Agregada Actual > ",len(samples), 20*"==","\n")
        # reportar casos con inconsistencia
        print(20*"==","Descartes en sanitizacion",len(sanitizer.trash), 20*"==","\n")
        print(20*"==","Duplicidad en sanitizacion",len(sanitizer.duplicated_keys), 20*"==","\n")

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
        #reporte de datos que no me sirver deacuerdo a mis criterios
        print(20*"==","Descartes en construccion",len(builder.trash), 20*"==","\n")
        if len(builder.trash) > 0:
            print(20*"==","Descartes en ejemplo: ", 20*"==","\n")
            print(builder.trash[np.random.randint(0,len(builder.trash))])
        print(20*"==","Duplicidad en construccion",len(builder.duplicated), 20*"==","\n")
        # exportar el dataset a un .json 
        export_to_json(dataset, './Exports/dataset_procesado.json')

        print(20*"==","Resumen de dataset", 20*"==","\n")
        for key, value in dataset.items():
            print(f"{key}: {len(value)} items")
        #separar en entrenamiento, validacion y pruebas
        trainer = builder.split_dataset()
        for key in trainer.keys():
            export_to_json(trainer[key],'./Exports/dataset_procesado_' + key + '.json')

        print(20*"==","Resumen de division (train, validation,test)", 20*"==","\n")
        for key, value in trainer.items():
            print(f"{key}: {len(value)} llaves")
            for key, value in value.items():
                print(f"{key}: {len(value)} items")
            print("---" * 40, "\n")
        
        #consultas
        query = Query(dataset)
        print(20*"==","Explorando datasets", 20*"==","\n")
        print(query.get_by_id('tra_099'))
        # filtrar por label
        query_by_label = query.filter_by_label('warning')
        print(20*"==","Comprobacion del filtro metadatos", 20*"==","\n")
        print("Unica etiqueta: ",set([ s['label'] for s in query_by_label ]), "\n")
        print(query_by_label[np.random.randint(0,len(query_by_label))])
        # print(query_by_label)
        # filtrar metadatos
        query_by_metadata = query.filter_by_metadata('crucero','cruce_d')
        # print(query_by_metadata)
        print(20*"==","Comprobacion del filtro metadatos", 20*"==","\n")
        print("Unico metadato: ",set([ s['metadata']['crucero'] for s in query_by_metadata ]), "\n")
        print(query_by_metadata[np.random.randint(0,len(query_by_metadata))])


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

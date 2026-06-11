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
from Core.Layer import *
from Core.Prediction import *


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
# exportar datasets a json
def export_to_json(dataset, name):
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4, cls=ExportJSON)
    print(20*"==","Dataset exportado como: ",name, 20*"==","\n")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Proyecto Final IA - Red Neuronal")
    parser.add_argument("csv_path", nargs="?", default='./Datasets/dataset_sucio_trafico_urbano_200_muestras.csv',
                        help="Ruta del archivo CSV de entrada")
    parser.add_argument("-m", "--model", help="Ruta del archivo .npz de modelo entrenado a cargar para inferencia directa")
    args = parser.parse_args()
    
    ruta_archivo = args.csv_path
        
    # cargamos el csv con la ruta
    contenido = load_csv(ruta_archivo)
    #si existe info hacemos el tratamiento
    if contenido:
        print(f"Se cargaron {len(contenido)} registros correctamente.")
        #sanilizar los datos
        id_string = 'sample_id'
        sanitizer = Sanitizer(contenido)
        #limpiar datos crudos 
        sanitizer.clean_data()
        #limpiar datos duplicados 
        sanitizer.remove_duplicates()
        # Dar formato a todas las columnas con un tipo de datos custom
            # la funcion parse_number quita las comas de un numero string (plus :D)
        sanitizer.format_columns({id_string:str,'vehiculos_hora':sanitizer.parse_number,'velocidad_promedio_kmh':sanitizer.parse_number,'densidad_vehicular':sanitizer.parse_number,'tiempo_espera_s':sanitizer.parse_number})
        #aqui ya tenemos los datos crudos buenos
        data = sanitizer.data
        # Exploarcion
        print(20*"==","Explorando muestras crudas", 20*"==","\n")
        print(data[0])
        # Aqui le damos forma a la estrucutra deseada [id, features,metadata,labels]
        samples = sanitizer.shaper(id_string,'estado', ['vehiculos_hora','velocidad_promedio_kmh','densidad_vehicular','tiempo_espera_s'], [ 'fecha_medicion', 'vehiculos_hora','crucero','zona','calidad_medicion','observacion'])

        # Primer elemento del samples para ver el procesamiento
        print(20*"==","Explorando muestras estructuradas", 20*"==","\n")
        first_sample = samples[0]
        print(first_sample)
        features_length = len(first_sample['features'])
        #agregar datos sinteticos a samples
        print(20*"==","Agregando muestra estructurada Actual > ",len(samples), 20*"==","\n")

        new_sample = {'id': 'tra_201', 'label': 'critical', 'features': [849.24, 20, 66.87, 178.8], 'metadata': {'fecha_medicion': '2026-04-14', 'vehiculos_hora': 849.24, 'crucero': 'cruce_c', 'zona': 'oriente', 'calidad_medicion': 'alta', 'observacion': 'que calor hace'}}
        samples  = sanitizer.add_sample(new_sample, features_length)
        print(20*"==","Agregada Actual > ",len(samples), 20*"==","\n")
        
        # reportar casos con inconsistencia
        print(20*"==","Descartes en sanitizacion",len(sanitizer.trash), 20*"==","\n")
        print(20*"==","Duplicidad en sanitizacion",len(sanitizer.duplicated_keys), 20*"==","\n")

        # Crear la clase dataset
        builder = Builder(samples)
        # Descartar valores con inconsistencias y descartarlos
        builder.evaluate_quality()
        # borrar duplicadidad en features o ids
        builder.avoid_duplicity()
        # llevar los labels a un estdo similar decodificando abreviaciones o lenguaje 
        builder.normalize_labels()
        # llevar metadatos que me importan a un estado similar de significado inlges -> esp
        builder.normalize_metadata('calidad_medicion')
        # guardar el dataset en una variable ya con la estructura buscad {x,y,id,metadata, cti,itc, samples}
        dataset = builder.build_dataset()
        #reporte de datos que no me sirver deacuerdo a mis criterios
        print(20*"==","Descartes en construccion",len(builder.trash), 20*"==","\n")
        if len(builder.trash) > 0:
            print(20*"==","Descartes en ejemplo: ", 20*"==","\n")
            print(builder.trash[np.random.randint(0,len(builder.trash))])
        print(20*"==","Duplicidad en construccion",len(builder.duplicated), 20*"==","\n")
        # exportar el dataset a un .json 
        export_to_json(dataset, './Exports/dataset_procesado.json')
        # evaluar con estadistica basica
        print(20*"==","Resumen de dataset", 20*"==","\n")
        for key, value in dataset.items():
            print(f"{key}: {len(value)} items")
        #separar en entrenamiento, validacion y pruebas
        trainer = builder.split_dataset()
        
        if args.model:
            print("\n" + 20*"==" + f" Modo Inferencia Directa: Cargando {args.model} " + 20*"==")
            from Core.Trainer import Trainer
            
            # Inicializar el entrenador (para usar sus métodos de evaluación y normalización)
            model_trainer = Trainer(train_data=trainer["train"])
            try:
                model_trainer.load_parameters(args.model)
            except Exception as e:
                print(f"Error al cargar el modelo desde {args.model}: {e}")
                import sys
                sys.exit(1)
                
            # Normalizar los datos del test set usando las estadísticas cargadas del modelo
            X_test_norm = (trainer["test"]["x"] - model_trainer.mean) / model_trainer.std
            y_pred_test, probs_test = model_trainer.predict(X_test_norm)
            
            n_classes = len(trainer["train"]["class_to_index"])
            
            print("\n" + 20*"==" + " Métricas en el Conjunto de Test " + 20*"==")
            print("Test accuracy: ", model_trainer.accuracy_score(trainer["test"]["y"], y_pred_test))
            
            # Matriz de Confusión
            cm = model_trainer.confusion_matrix(trainer["test"]["y"], y_pred_test, n_classes=n_classes)
            print("\nMatriz de confusión (Test):")
            print("Rows = real class; columns = predicted class")
            print(cm)
            
            # Mostrar primeras predicciones del conjunto de prueba
            print("\nPrimeras predicciones (Test):")
            for i in range(min(10, len(trainer["test"]["y"]))):
                print(
                    trainer["test"]["ids"][i],
                    "real=", trainer["test"]["index_to_class"][int(trainer["test"]["y"][i])],
                    "predicted=", trainer["test"]["index_to_class"][int(y_pred_test[i])],
                    "probs=", np.round(probs_test[i], 3)
                )
                
            # Reporte detallado de errores (predicciones incorrectas)
            errors = model_trainer.error_report(
                trainer["test"]["ids"],
                trainer["test"]["y"],
                y_pred_test,
                probs_test,
                trainer["test"]["index_to_class"]
            )
            print(f"\nErrores detectados en test: {len(errors)}")
            for item in errors[:5]:
                print(item)
                
            # Compatibilidad con el esquema original de Prediction y Layer
            print("\n" + 20*"==" + " Verificación de Compatibilidad (Prediction & Layer) " + 20*"==")
            trained_layer1 = model_trainer.layers[0]
            trained_layer2 = model_trainer.layers[1]
            
            prediction = Prediction(dataset)
            prediction.addLayer(trained_layer1)
            prediction.addLayer(trained_layer2)
            print(f"Cargadas {len(prediction.layers)} capas en Prediction.")
            
            # Normalizar X del dataset completo para que coincida con las capas entrenadas
            mean_dataset = dataset["x"].mean(axis=0)
            std_dataset = dataset["x"].std(axis=0)
            std_dataset = np.where(std_dataset == 0, 1.0, std_dataset)
            prediction.X = (dataset["x"] - mean_dataset) / std_dataset
            
            # Inferencia individual secuencial
            prediction.predict(layer_index1=0, layer_index2=1)
            
            random_idx = np.random.randint(0, len(prediction.predictions))
            print(f"\nPredicción aleatoria con Prediction original (muestra índice {random_idx}):")
            for key, value in prediction.predictions[random_idx].items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {np.round(value, 4)}")
                else:
                    print(f"  {key}: {value}")
                    
            # Finalizar la ejecución
            import sys
            sys.exit(0)
        for key in trainer.keys():
            export_to_json(trainer[key],'./Exports/dataset_procesado_' + key + '.json')
        #evaluar si la separacion se hizo bien
        print(20*"==","Resumen de division (train, validation,test)", 20*"==","\n")
        for key, value in trainer.items():
            print(f"{key}: {len(value)} llaves")
            for key, value in value.items():
                print(f"{key}: {len(value)} items")
            print("---" * 40, "\n")
        
        #consultas
        query = Query(dataset)
        #se construye el indexado -> sample al inicial la clase
        print(20*"==","Explorando datasets", 20*"==","\n")
        #busqueda rapida por id
        print(query.get_by_id('tra_099'))
        # filtrar por label
        query_by_label = query.filter_by_label('warning')
        print(20*"==","Comprobacion del filtro label", 20*"==","\n")
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
        
        
        print(20*"==","Entrenamiento y Evaluación de la Red Neuronal", 20*"==","\n")
        from Core.Trainer import Trainer
        
        # Inicializar el entrenador con los conjuntos de entrenamiento y validación
        model_trainer = Trainer(train_data=trainer["train"], val_data=trainer["validation"])
        
        # 1. Normalización para pruebas rápidas y gradient check
        mean_train = trainer["train"]["x"].mean(axis=0)
        std_train = trainer["train"]["x"].std(axis=0)
        std_train = np.where(std_train == 0, 1.0, std_train)
        X_train_n = (trainer["train"]["x"] - mean_train) / std_train
        y_train = trainer["train"]["y"]
        
        n_features = X_train_n.shape[1]
        n_classes = len(trainer["train"]["class_to_index"])
        
        # 2. Inicializar la red neuronal e imprimir dimensiones
        model_trainer.initialize_network(n_features=n_features, n_hidden=8, n_classes=n_classes, seed=42)
        print("Dimensiones de los parámetros inicializados:")
        for idx, layer in enumerate(model_trainer.layers):
            print(f"Capa {idx+1} - Pesos W: {layer.weights.shape}, Sesgo b: {layer.bias.shape}")
            
        # 3. Verificación Numérica de Gradientes (Gradient Checking)
        num, ana, diff = model_trainer.numerical_gradient_check(X_train_n[:10], y_train[:10])
        print(f"\nGradient check[0,0] -> numeric = {num:.8f}, analitic={ana:.8f}, diference={diff:.8e}")
        
        # 4. Entrenamiento completo de la red neuronal
        print("\nEntrenando la red neuronal...")
        params_trained, history = model_trainer.train(
            n_hidden=8,
            lr=0.03,
            epochs=600,
            seed=42
        )
        
        # 5. Evaluación del modelo entrenado en todas las particiones
        X_train_norm = (trainer["train"]["x"] - model_trainer.mean) / model_trainer.std
        X_val_norm = (trainer["validation"]["x"] - model_trainer.mean) / model_trainer.std
        X_test_norm = (trainer["test"]["x"] - model_trainer.mean) / model_trainer.std
        
        y_pred_train, _ = model_trainer.predict(X_train_norm)
        y_pred_val, _ = model_trainer.predict(X_val_norm)
        y_pred_test, probs_test = model_trainer.predict(X_test_norm)
        
        print("\n" + 20*"==" + " Métricas Finales " + 20*"==")
        print("Train accuracy: ", model_trainer.accuracy_score(trainer["train"]["y"], y_pred_train))
        print("Validation accuracy: ", model_trainer.accuracy_score(trainer["validation"]["y"], y_pred_val))
        print("Test accuracy: ", model_trainer.accuracy_score(trainer["test"]["y"], y_pred_test))
        
        # 6. Matriz de Confusión
        cm = model_trainer.confusion_matrix(trainer["test"]["y"], y_pred_test, n_classes=n_classes)
        print("\nMatriz de confusión (Test):")
        print("Rows = real class; columns = predicted class")
        print(cm)
        
        # 7. Mostrar primeras predicciones del conjunto de prueba
        print("\nPrimeras predicciones (Test):")
        for i in range(min(10, len(trainer["test"]["y"]))):
            print(
                trainer["test"]["ids"][i],
                "real=", trainer["test"]["index_to_class"][int(trainer["test"]["y"][i])],
                "predicted=", trainer["test"]["index_to_class"][int(y_pred_test[i])],
                "probs=", np.round(probs_test[i], 3)
            )
            
        # 8. Reporte detallado de errores (predicciones incorrectas)
        errors = model_trainer.error_report(
            trainer["test"]["ids"],
            trainer["test"]["y"],
            y_pred_test,
            probs_test,
            trainer["test"]["index_to_class"]
        )
        print(f"\nErrores detectados en test: {len(errors)}")
        for item in errors[:5]:
            print(item)
            
        # 9. Experimentos guiados con hiperparámetros de la Actividad 6
        random_seed = np.random.randint(1000)
        experiments = [
            {"n_hidden": 8, "lr": 0.003, "epochs": 600, "seed": random_seed},
            {"n_hidden": 8, "lr": 0.03, "epochs": 600, "seed": random_seed},
            {"n_hidden": 8, "lr": 0.3, "epochs": 600, "seed": random_seed},
            {"n_hidden": 16, "lr": 0.03, "epochs": 600, "seed": random_seed},
        ]
        results = []
        print("\n" + 20*"==" + " Laboratorio de Experimentos Modular (Actividad 6) " + 20*"==")
        
        # Muestra aleatoria de test para predecir
        random_index = np.random.randint(len(trainer["test"]["y"]))
        X_test_to_predict = trainer["test"]["x"][random_index]
        y_test_to_predict = trainer["test"]["y"][random_index]
        
        for index, experiment in enumerate(experiments):
            print(f"\nEjecutando Experimento {index}: {experiment}")
            exp_trainer = Trainer(train_data=trainer["train"], val_data=trainer["validation"])
            
            # Usar la función modular de correr experimentos
            experiment_name, metrics, params_trained, history = exp_trainer.run_experiment(experiment, index)
            
            # Guardar el modelo del experimento en Exports
            model_path = f"./Exports/{experiment_name}"
            exp_trainer.save_parameters(model_path)
            
            # Cargar el modelo guardado para validación de inferencia
            loaded_exp_trainer = Trainer(train_data=trainer["train"])
            loaded_exp_trainer.load_parameters(model_path)
            
            # Predecir sobre la muestra aleatoria seleccionada
            x_sample = np.array(X_test_to_predict, dtype=np.float32).reshape(1, -1)
            x_sample_norm = (x_sample - loaded_exp_trainer.mean) / loaded_exp_trainer.std
            pred_idx, probs_new = loaded_exp_trainer.predict(x_sample_norm)
            pred_label = trainer["train"]["index_to_class"][pred_idx[0]]
            
            print(f"  Inferencia sobre muestra aleatoria - Clase Real: {trainer['train']['index_to_class'][int(y_test_to_predict)]} | Clase Predicha: {pred_label}")
            print(f"  Probabilidades: {np.round(probs_new[0], 4)}")
            
            # Visualizar y guardar los resultados del historial del experimento
            curve_name_loss, curve_name_accuracy = exp_trainer.plot_history(
                filename_loss=f"./Exports/{experiment_name.replace('.npz', '')}-loss_curve.png",
                filename_acc=f"./Exports/{experiment_name.replace('.npz', '')}-accuracy_curve.png",
                experiment_name=experiment_name
            )
            
            results.append({
                "metrics": metrics,
                "filename": experiment_name,
                "experiment": experiment,
                "curve_name_loss_image": curve_name_loss,
                "curve_name_accuracy_image": curve_name_accuracy,
                "predicted_label": pred_label,
                "real_label": trainer["train"]["index_to_class"][int(y_test_to_predict)],
                "probabilities": np.round(probs_new[0], 4).tolist()
            })
            
        # Resumen de resultados: guardar experimentos y ubicaciones de los modelos en un JSON
        with open("./Exports/results.json", "w", encoding="utf-8") as outfile:
            json.dump(results, outfile, indent=4, ensure_ascii=False)
            
        print("\n" + 20*"==" + " Guardado de Parámetros de Modelo Base " + 20*"==")
        # 10. Guardar y graficar el modelo base inicial
        model_trainer.save_parameters('./Exports/modelo_entrenado.npz')
        print("Modelo base inicial guardado en: './Exports/modelo_entrenado.npz'")
        
        # 11. Graficación del historial de entrenamiento base
        model_trainer.plot_history(
            filename_loss='./Exports/loss_curve.png',
            filename_acc='./Exports/accuracy_curve.png',
            experiment_name="Base_Model"
        )
        print("Gráficas del modelo base guardadas como './Exports/loss_curve.png' y './Exports/accuracy_curve.png'")
        print("Todos los resultados de experimentos guardados en './Exports/results.json'")
        
        # 12. Compatibilidad con el esquema original de Prediction y Layer
        print("\n" + 20*"==" + " Verificación de Compatibilidad (Prediction & Layer) " + 20*"==")
        trained_layer1 = model_trainer.layers[0]
        trained_layer2 = model_trainer.layers[1]
        
        prediction = Prediction(dataset)
        prediction.addLayer(trained_layer1)
        prediction.addLayer(trained_layer2)
        print(f"Cargadas {len(prediction.layers)} capas en Prediction.")
        
        # Normalizar X del dataset completo para que coincida con las capas entrenadas
        mean_dataset = dataset["x"].mean(axis=0)
        std_dataset = dataset["x"].std(axis=0)
        std_dataset = np.where(std_dataset == 0, 1.0, std_dataset)
        prediction.X = (dataset["x"] - mean_dataset) / std_dataset
        
        # Inferencia individual secuencial simulando la estructura del proyecto original
        prediction.predict(layer_index1=0, layer_index2=1)
        
        random_idx = np.random.randint(0, len(prediction.predictions))
        print(f"\nPredicción aleatoria con Prediction original (muestra índice {random_idx}):")
        for key, value in prediction.predictions[random_idx].items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {np.round(value, 4)}")
            else:
                print(f"  {key}: {value}")


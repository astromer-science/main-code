import os
import glob
import toml
import re
import pandas as pd
from io import StringIO

def read_toml(path):
    """
        Lee y parsea un archivo TOML (Tom's Obvious, Minimal Language).

        Args:
            ruta_archivo: La ruta completa o relativa al archivo .toml.

        Returns:
            Un diccionario (Dict) de Python con el contenido del archivo TOML.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            toml.TomlDecodeError: Si el contenido del archivo TOML no es válido.
        """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            metrics = toml.load(f)
        return metrics
    except FileNotFoundError:
        print(f"Error: El archivo no fue encontrado en la ruta '{path}'.")
        raise
    except toml.TomlDecodeError as e:
        print(f"Error: El archivo TOML no es válido. Detalles: {e}")
        raise

def procesar_metricas(paths_simulados, folder='output'):
    """
    Procesa una lista de rutas de archivos TOML simulados y retorna un DataFrame.
    (Se mantiene la simulación de lectura del TOML y la extracción de parámetros del path)
    """
    datos_tabla = []
    models = paths_simulados
    
    patron = re.compile(rf'^\./{re.escape(folder)}/clf_([a-z]+)_(\d+)_(\d+)/([a-z_]+)/test_metrics\.toml$')
    for path in models:
        match = patron.match(path)

        if match:
            dataset_name = match.group(1)
            fold = int(match.group(2))
            num_samples = int(match.group(3))
            classifier_name = match.group(4)
            metricas = read_toml(path) 
            
            test_acc = float(metricas.get('test_acc', 0.0))
            test_precision = float(metricas.get('test_precision', 0.0))
            test_recall = float(metricas.get('test_recall', 0.0))
            test_f1 = float(metricas.get('test_f1', 0.0))

            datos_tabla.append({
                'Dataset': dataset_name,
                'Fold': fold,
                'Samples': num_samples,
                'Classifier': classifier_name,
                'Accuracy': test_acc,   # Debe ser 'Accuracy'
                'Precision': test_precision, # Debe ser 'Precision'
                'Recall': test_recall,   # Debe ser 'Recall'
                'F1_Score': test_f1     # Debe ser 'F1_Score'
                })

    return pd.DataFrame(datos_tabla)

def summarize_metrics_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa el DataFrame de métricas por Classifier, Dataset y Samples, 
    y calcula la media, percentil 95 y percentil 5 para las métricas.
    """
    group_cols = ['Classifier', 'Dataset', 'Samples']
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    
    # 1. Definir las agregaciones usando una lista de tuplas para la consistencia
    agg_list = [
        ('mean', 'mean'), 
        ('p95', lambda x: x.quantile(0.95)), 
        ('p05', lambda x: x.quantile(0.05))
    ]

    # Crear el diccionario de agregación: {'Accuracy': [('mean', 'mean'), ...], ...}
    agg_dict = {
        col: agg_list for col in metric_cols
    }

    # Aplicar la agrupación y las funciones de agregación
    df_agrupado = df.groupby(group_cols)[metric_cols].agg(agg_dict)

    # 2. Aplanar los nombres de las columnas
    # El MultiIndex es: (Métrica, Función)
    df_agrupado.columns = [f'{col[0]}_{col[1]}' for col in df_agrupado.columns.values]
    df_final = df_agrupado.reset_index()

    # 3. Crear una columna de presentación con el formato "Media [P05, P95]"
    columnas_finales = group_cols.copy()
    for metric in metric_cols:
        # Accediendo a los nombres de las columnas aplanadas: 'Accuracy_mean', 'Accuracy_p05', etc.
        mean_col = f'{metric}_mean'
        p05_col = f'{metric}_p05'
        p95_col = f'{metric}_p95'
        
        # Combinar los valores en una sola columna legible
        df_final[f'{metric} (Mean [P05, P95])'] = df_final.apply(
            lambda row: f"{row[mean_col]:.4f} [{row[p05_col]:.4f}, {row[p95_col]:.4f}]", 
            axis=1
        )
        columnas_finales.append(f'{metric} (Mean [P05, P95])')

    return df_final[columnas_finales]

def summarize_metrics_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa el DataFrame de métricas y calcula la media, percentil 95 y percentil 5.
    Retorna el DataFrame con columnas separadas para graficar.
    """
    group_cols = ['Classifier', 'Dataset', 'Samples']
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    
    agg_list = [
        ('mean', 'mean'), 
        ('p95', lambda x: x.quantile(0.95)), 
        ('p05', lambda x: x.quantile(0.05))
    ]

    agg_dict = {
        col: agg_list for col in metric_cols
    }

    df_agrupado = df.groupby(group_cols)[metric_cols].agg(agg_dict)
    df_agrupado.columns = [f'{col[0]}_{col[1]}' for col in df_agrupado.columns.values]
    df_final = df_agrupado.reset_index()

    # Crear una columna de faceta combinada para Dataset y Samples
    df_final['Dataset_Samples'] = df_final['Dataset'] + ' (' + df_final['Samples'].astype(str) + ' samples)'
    
    return df_final
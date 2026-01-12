import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import toml
import re
import os

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
    Procesa una lista de rutas de archivos TOML y retorna un DataFrame.
    """
    datos_tabla = []
    
    folder_clean = folder.rstrip('/')
    
    # Escapamos los caracteres especiales (como el punto de ./)
    folder_escaped = re.escape(folder_clean)

    patron = re.compile(rf'^{folder_escaped}/clf_([a-z]+)_(\d+)_(\d+)/([a-z_]+)/test_metrics\.toml$')

    print(f"[DEBUG] Patrón Regex usado: {patron.pattern}") # Útil para verificar

    for path in paths_simulados:
        match = patron.match(path)
        
        if match:
            dataset_name = match.group(1)
            fold = int(match.group(2))
            num_samples = int(match.group(3))
            classifier_name = match.group(4)
            
            # Leemos el archivo real (asegúrate de que read_toml esté importado o definido)
            # Si paths_simulados son rutas reales, usa toml.load
            try:
                metricas = toml.load(path) # O tu función read_toml(path)
                
                test_acc = float(metricas.get('test_acc', 0.0))
                test_precision = float(metricas.get('test_precision', 0.0))
                test_recall = float(metricas.get('test_recall', 0.0))
                test_f1 = float(metricas.get('test_f1', 0.0))

                datos_tabla.append({
                    'Dataset': dataset_name,
                    'Fold': fold,
                    'Samples': num_samples,
                    'Classifier': classifier_name,
                    'Accuracy': test_acc,
                    'Precision': test_precision,
                    'Recall': test_recall,
                    'F1_Score': test_f1
                })
            except Exception as e:
                print(f"[ERROR] No se pudo leer {path}: {e}")
        else:
            # Opcional: ver qué archivos fallan
            # print(f"[WARN] No matcheó: {path}")
            pass

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

def plot_classification_metrics(df, fig, axes, dataset_name):
    """
    Genera el gráfico de barras. Si recibe datos crudos (por fold),
    calcula automáticamente la media y los errores (min/max).
    """
    
    # --- 1. PRE-PROCESAMIENTO Y AGREGACIÓN ---
    df_plot = df.copy()
    
    # Normalizar nombres
    df_plot['Dataset'] = df_plot['Dataset'].replace({'alcock': 'Alcock', 'atlas': 'ATLAS'})
    
    # VERIFICACIÓN CLAVE: Si no existen las columnas de error, las calculamos
    required_cols = ['F1_Score_mean', 'F1_Score_error_low', 'F1_Score_error_high']
    if not all(col in df_plot.columns for col in required_cols):
        print("[INFO] Calculando medias y errores (min/max) a partir de los folds...")
        
        # Agrupamos por Dataset, Samples y Classifier para colapsar los folds
        # Calculamos media, min y max del F1_Score
        agg_df = df_plot.groupby(['Dataset', 'Samples', 'Classifier'])['F1_Score'].agg(['mean', 'min', 'max']).reset_index()
        
        # Creamos las columnas que el plot espera
        agg_df['F1_Score_mean'] = agg_df['mean']
        # Error hacia abajo: Media - Mínimo
        agg_df['F1_Score_error_low'] = agg_df['mean'] - agg_df['min']
        # Error hacia arriba: Máximo - Media
        agg_df['F1_Score_error_high'] = agg_df['max'] - agg_df['mean']
        
        # Reemplazamos df_plot con la versión agregada
        df_plot = agg_df

    # --- 2. CONFIGURACIÓN DE NOMBRES Y ESTILOS ---
    name_mapping = {
        'avg': 'A2 + Avg Pool', 
        'max': 'A2 + Max Pool', 
        'skip': 'A2 + Skip Conn',
        'att_avg': 'Attn (Avg)', 
        'att_cls': 'Attn (CLS)',
        'base_avgpool': 'Base (MLP)', 
        'base_gru': 'Base (GRU)', 
        'raw_gru': 'Base (Raw RNN)'
    }
    
    if 'Classifier' in df_plot.columns:
        df_plot['Display_Name'] = df_plot['Classifier'].replace(name_mapping)
    else:
        df_plot['Display_Name'] = df_plot.index

    STYLE_COLORS = {
        'bar_face': '#B0E3E6',
        'bar_edge': '#10739E',
        'error_bar': '#AE4132'
    }

    # Filtrar dataset
    df_dataset = df_plot[df_plot['Dataset'].str.lower() == dataset_name.lower()].copy()
    
    if df_dataset.empty:
        print(f"[ADVERTENCIA] No hay datos para el dataset: {dataset_name}")
        return

    unique_samples = sorted(df_dataset['Samples'].unique())
    
    if len(unique_samples) > len(axes):
        print(f"[ERROR] Samples ({len(unique_samples)}) > Ejes ({len(axes)})")
        return
    
    # --- 3. PLOTTING ---
    for k, spc in enumerate(unique_samples):
        ax = axes[k]
        
        # Filtramos para este SPC
        df_subset = df_dataset[df_dataset['Samples'] == spc].copy()
        
        # Ordenar
        df_subset = df_subset.sort_values(by='F1_Score_mean', ascending=True)
        
        # Datos X e Y
        x_pos = np.arange(len(df_subset))
        means = df_subset['F1_Score_mean'].values
        labels = df_subset['Display_Name'].values
        labels_multiline = [str(name).replace(' ', '\n') for name in labels]
        
        # Config Ejes
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_multiline, rotation=0, ha='center', fontsize=12)
        
        # Errores para matplotlib (shape 2xN)
        errors = [
            df_subset['F1_Score_error_low'].values, 
            df_subset['F1_Score_error_high'].values
        ]
        
        # Barras
        ax.bar(x_pos, means, yerr=errors, 
               color=STYLE_COLORS['bar_face'], 
               edgecolor=STYLE_COLORS['bar_edge'], 
               ecolor=STYLE_COLORS['error_bar'], 
               capsize=3, linewidth=1.2)
        
        ax.set_title('{} samples per class'.format(spc), fontsize=16)
        ax.set_ylim(0, 1.15) 
        
        # Anotaciones
        for i, value in enumerate(means):
            std_val = errors[1][i] # Usamos el error superior para el texto (±Max)
            text_y_pos = value + std_val + 0.02
            label_text = '{:.1f}%\n(±{:.2f})'.format(value*100, std_val)
            
            ax.text(i, text_y_pos, label_text, 
                    ha='center', va='bottom', fontsize=12, color='black',
                    linespacing=1.2)
            
    axes[0].set_ylabel('F1-Score', fontsize=16)
    
    display_ds_name = df_dataset['Dataset'].iloc[0]
    fig.suptitle(f'Classification on {display_ds_name}', y=1.05, fontsize=16)
    return fig, axes
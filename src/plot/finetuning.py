import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from glob import glob

def parse_path_info(file_path):
    try:
        abs_path = os.path.abspath(file_path)
        parts = abs_path.split(os.sep)
        exp_folder = parts[-2]   # Ej: alcock_20
        fold_folder = parts[-3]  # Ej: fold_0
        
        # Extraer SPC y Dataset desde el nombre de la carpeta del experimento
        # Ej: "alcock_20" -> dataset="alcock", spc=20
        if '_' in exp_folder:
            spc = int(exp_folder.split('_')[-1])
            dataset_derived = "_".join(exp_folder.split('_')[:-1])
        else:
            spc = -1
            dataset_derived = exp_folder

        # Extraer número de Fold
        # Ej: "fold_0" -> 0
        if 'fold' in fold_folder:
            fold_num = int(fold_folder.split('_')[-1])
        else:
            fold_num = -1

        return dataset_derived, fold_num, spc
        
    except Exception as e:
        print(f"[WARN] No se pudo parsear la ruta {file_path}: {e}")
        return "unknown", -1, -1

def extract_ft_metrics(root_path):
    """
    Recorre recursivamente root_path buscando archivos de resultados
    y los consolida en un único DataFrame.
    """
    print(f"[INFO] Buscando resultados en: {root_path}")
    
    # Patrón para encontrar tanto results_custom.csv como results_base_custom.csv
    # search_pattern = os.path.join(root_path, "**", "results*_custom.csv")
    # Usamos glob recursivo
    files = glob(f"{root_path}/**/results*_custom.csv", recursive=True)
    
    if not files:
        print("[ERROR] No se encontraron archivos CSV que coincidan con el patrón.")
        return pd.DataFrame()

    all_dfs = []

    for f in files:
        try:
            # 1. Leer el CSV
            df = pd.read_csv(f)
            
            # 2. Determinar si es Base o Finetuned por el nombre del archivo
            filename = os.path.basename(f)
            if 'base' in filename:
                model_stage = 'base' # Pre-trained (sin finetune)
            else:
                model_stage = 'finetuned' # Post-finetune
            
            # 3. Extraer info de la ruta
            dataset_name, fold_num, spc = parse_path_info(f)
            
            # 4. Agregar columnas de metadatos al DF
            df['experiment_id'] = dataset_name + "_" + str(spc)
            df['dataset_name'] = dataset_name
            df['fold'] = fold_num
            df['spc'] = spc
            df['model_stage'] = model_stage
            df['source_file'] = f
            
            # Si 'training_time' no existe (ej. en el base), rellenar con 0 o NaN
            if 'training_time' not in df.columns:
                df['training_time'] = 0.0
                
            all_dfs.append(df)
            
        except Exception as e:
            print(f"[ERROR] Falló al leer {f}: {e}")

    # 5. Concatenar todo
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Reordenar columnas para que las importantes salgan primero
        cols_order = ['dataset_name', 'spc', 'fold', 'model_stage', 'test_mse', 'test_r2', 'training_time']
        # Agregar el resto de columnas que existan
        remaining_cols = [c for c in final_df.columns if c not in cols_order]
        final_df = final_df[cols_order + remaining_cols]
        
        # Ordenar filas
        final_df = final_df.sort_values(by=['dataset_name', 'spc', 'fold', 'model_stage'])
        
        return final_df
    else:
        return pd.DataFrame()
    
def plot_ft(df, fig, axes):
    """
    Genera el gráfico de barras con doble eje Y en los ejes proporcionados
    con una leyenda única global en la parte superior.
    """
    # 1. Preparación de Datos
    plot_df = df.copy()
    plot_df = plot_df[['dataset_name', 'spc', 'model_stage', 'test_mse', 'training_time']]
    
    # Renombrar etapas
    plot_df['model_stage'] = plot_df['model_stage'].replace({
        'base': 'Pretrained', 
        'finetuned': 'Finetuned'
    })
    # Convertir tiempo a minutos
    plot_df['training_time_min'] = plot_df['training_time'] / 60.0

    # Títulos y Datasets
    titles = {'alcock': 'Alcock', 'atlas': 'ATLAS'}
    datasets = ['alcock', 'atlas']
    
    # Colores
    color_pre = '#AE4132'   # Rojo oscuro (Pretrained)
    color_fine = '#49a3a9'  # Turquesa (Finetuned)
    color_time = '#fad7ac'  # Beige/Amarillo (Tiempo)
    
    # Ancho de las barras
    width = 0.25 
    
    # Validación básica
    if len(axes) < len(datasets):
        print(f"[ERROR] Se requieren al menos {len(datasets)} ejes, se recibieron {len(axes)}.")
        return fig

    # Variables para guardar los handles/labels de la leyenda
    global_handles = []
    global_labels = []

    for i, ds in enumerate(datasets):
        ax1 = axes[i]
        subset = plot_df[plot_df['dataset_name'] == ds]
        
        # Si no hay datos para este dataset, saltar
        if subset.empty:
            continue
        
        # 2. Agregación Manual (Media, Min, Max)
        agg = subset.groupby(['spc', 'model_stage']).agg(
            rmse_mean=('test_mse', 'mean'),
            rmse_min=('test_mse', 'min'),
            rmse_max=('test_mse', 'max'),
            time_mean=('training_time_min', 'mean'),
            time_min=('training_time_min', 'min'),
            time_max=('training_time_min', 'max')
        ).reset_index()

        # Eje X
        spcs = sorted(subset['spc'].unique())
        x = np.arange(len(spcs))
        
        # Separar datos
        pre_data = agg[agg['model_stage'] == 'Pretrained'].set_index('spc').reindex(spcs)
        fine_data = agg[agg['model_stage'] == 'Finetuned'].set_index('spc').reindex(spcs)
        
        # Calcular errores y graficar
        if not pre_data.empty:
            yerr_pre = np.array([
                pre_data['rmse_mean'] - pre_data['rmse_min'],
                pre_data['rmse_max'] - pre_data['rmse_mean']
            ])
            # Barra 1: Pretrained
            ax1.bar(x - width, pre_data['rmse_mean'], width, label='RMSE Pretrained', 
                    color='#fad9d5', 
                    yerr=yerr_pre, 
                    capsize=5, 
                    edgecolor='#AE4132',
                    alpha=0.6, hatch='\\\ ')

        if not fine_data.empty:
            yerr_fine = np.array([
                fine_data['rmse_mean'] - fine_data['rmse_min'],
                fine_data['rmse_max'] - fine_data['rmse_mean']
            ])
            yerr_time = np.array([
                fine_data['time_mean'] - fine_data['time_min'],
                fine_data['time_max'] - fine_data['time_mean']
            ])
            
            # Barra 2: Finetuned RMSE
            ax1.bar(x, fine_data['rmse_mean'], width, label='RMSE Finetuned', 
                    color='#b0e3e6', 
                    yerr=yerr_fine, 
                    capsize=6, 
                    edgecolor='#409fa5',
                    linewidth=1,
                    hatch='///')
            
            # --- EJE DERECHO (TIEMPO) ---
            ax2 = ax1.twinx()
            
            # Barra 3: Tiempo
            ax2.bar(x + width, fine_data['time_mean'], width, label='Finetuning Time', 
                    color=color_time, alpha=0.8, yerr=yerr_time, capsize=5, edgecolor='#be761d')
            
            ax2.set_ylabel('Time (minutes)', fontsize=14, color='k')
            ax2.tick_params(axis='y', labelcolor='k')
            # Limite dinamico
            ax2.set_ylim(0, agg['time_max'].max()+0.1)

            # --- CAPTURA DE LEYENDA (Solo la primera vez que tengamos datos) ---
            if not global_handles:
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                global_handles = h1 + h2
                global_labels = l1 + l2

        # Configuración Eje Izquierdo
        ax1.set_ylabel('RMSE', fontsize=14, color='black')
        ax1.set_xlabel('Samples per Class', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(spcs, fontsize=11)
        ax1.set_title(titles.get(ds, ds), fontsize=16)
        
        # Limite Y izquierdo
        if not agg.empty:
            ax1.set_ylim(0, agg['rmse_max'].max() * 1.2)
            
        ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # --- LEYENDA GLOBAL FUERA DEL BUCLE ---
    if global_handles:
        fig.legend(global_handles, global_labels, 
                   loc='upper center', 
                   bbox_to_anchor=(0.5, 1.07), # Ajusta este valor (1.08) para subir/bajar la leyenda
                   ncol=2, # Para que aparezcan en una sola fila horizontal
                   fontsize=12,
                   frameon=True) # Quitar el recuadro si prefieres limpieza

    return fig
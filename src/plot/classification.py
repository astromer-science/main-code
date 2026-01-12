import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 

def aggregate_classification_results(baseline_paths, experiment_paths, experiment_labels, metric_loader_func, spc_list=[20, 100, 500]):
    """
    Aggregates classification metrics from baseline definitions and new experiments.

    Args:
        baseline_paths (list of dict): List of dicts defining baselines. 
                                       e.g. [{'path': '...', 'label': 'A1', 'arch': 'avg_mlp'}]
        experiment_paths (list): List of paths for the new experiments.
        experiment_labels (list): List of labels corresponding to experiment_paths.
        metric_loader_func (callable): The function 'classification_metrics' (must accept path, spc_list, clf_arch).
        spc_list (list): List of SPC values.

    Returns:
        pd.DataFrame: A concatenated DataFrame containing all results.
    """
    df_list = []

    # 1. Load Baselines (Static comparisons like Donoso et al., A1, etc.)
    for base in baseline_paths:
        try:
            df = metric_loader_func(base['path'], spc_list=spc_list, clf_arch=base['arch'])
            # Create a column with the label repeated for all rows
            df['label'] = [base['label']] * df.shape[0]
            df_list.append(df)
        except Exception as e:
            print(f"Error loading baseline {base['label']}: {e}")

    # 2. Load New Experiments (Iterating over sorted_path)
    for path, label in zip(experiment_paths, experiment_labels):
        try:
            # Note: Your snippet did os.path.join(root, '..'). Adjust if necessary.
            target_path = os.path.join(path, '..') 
            df = metric_loader_func(target_path, spc_list=spc_list, clf_arch='skip_avg_mlp')
            df['label'] = [label] * df.shape[0]
            df_list.append(df)
        except Exception as e:
            print(f"Error loading experiment {label}: {e}")

    # 3. Concatenate and Clean
    if not df_list:
        return pd.DataFrame()
        
    final_df = pd.concat(df_list)
    final_df = final_df[~final_df['mean'].isna()]
    
    # Standardize labels if needed (optional cleanup from your snippet)
    final_df['label'] = final_df['label'].replace({'v0': 'A1', 'v1': 'A2'})
    
    return final_df

def _init_f1_layout(figsize=(6, 2)):
    """
    Initializes a 1x3 subplot layout for SPC comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True, dpi=300, 
                             gridspec_kw={'wspace': 0.07})
    return fig, axes

def plot_f1_scores(df, dataset_name, ylim=None, fig=None, axes=None):
    """
    Plots F1 Score bar charts for a specific dataset, split by SPC.

    Args:
        df (pd.DataFrame): DataFrame filtered for a specific dataset (e.g., only 'alcock').
        dataset_name (str): Name of the dataset for the title/filename (e.g., 'alcock').
        ylim (tuple, optional): (min, max) for Y-axis. Defaults to (0, 1).
        fig (Figure, optional): Existing Figure.
        axes (Axes, optional): Existing Axes array (1x3).
    """
    
    # --- Initialization ---
    if fig is None or axes is None:
        fig, axes = _init_f1_layout()
        
    # Style Config
    bar_color = '#B0E3E6'
    err_color = '#AE4132'
    edge_color = '#10739E'
    
    # --- Plotting Loop (Group by SPC) ---
    # We explicitly sort by SPC to ensure order 20 -> 100 -> 500
    spc_groups = df.groupby('spc')
    
    # Mapping SPC to axes index (assuming 20, 100, 500 order)
    unique_spcs = sorted(df['spc'].unique())
    
    for k, spc in enumerate(unique_spcs):
        if spc not in spc_groups.groups:
            continue
            
        group_df = spc_groups.get_group(spc)
        ax = axes[k]

        # Data preparation
        labels = group_df['label'].values
        means = group_df['mean'].values
        stds = group_df['std'].values
        x_pos = range(len(labels))
        
        # Draw Bars
        ax.bar(x_pos, means, yerr=stds, 
               color=bar_color, ecolor=err_color, edgecolor=edge_color)
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=0, fontsize=10)
        ax.set_title(f'{spc} samples\nper class', fontsize=10)
        
        if ylim:
            ax.set_ylim(ylim)
        
        # Text Annotations (Percentage above bar)
        for i, value in enumerate(means):
            # Dynamic offset based on plot index (from your snippet logic)
            offset = 0.04 if k == 0 else 0.02
            
            # Format as percentage
            text_str = '{:.1f}%'.format(value * 100)
            
            ax.text(i - 0.4, value + offset, text_str, fontsize=8, rotation=0)

    # --- Global Styling ---
    axes[0].set_ylabel('Test F1 Score', fontsize=12)
    
    return fig, axes
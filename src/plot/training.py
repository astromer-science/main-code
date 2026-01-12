# src/plot/training.py
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def _init_learning_curve_layout(figsize=(4, 3.5)):
    """
    Initializes the specific 2x1 layout for learning curves.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=300, sharex=True, 
                             gridspec_kw={'wspace': 0.2, 'hspace': 0.1})
    return fig, axes

def plot_learning_curves(train_metrics, val_metrics, paths, labels, fig=None, axes=None):
    """
    Plots Loss and R-square learning curves with Test set benchmarks.

    Args:
        train_metrics (list): List of training DataFrames.
        val_metrics (list): List of validation DataFrames.
        paths (list): List of paths to find 'results.csv'.
        labels (list): List of label strings for the legend.
        fig (Figure, optional): Existing Figure.
        axes (Axes, optional): Existing Axes array.
    """
    
    # --- Initialization ---
    if fig is None or axes is None:
        fig, axes = _init_learning_curve_layout()
        
    # Style constants from your snippet
    colormap = ['k', 'gray', 'b', 'g', 'y']
    train_color = '#9673A6'
    
    # --- Main Loop ---
    for k, path in enumerate(paths):
        # Safety check for lists length
        if k >= len(train_metrics) or train_metrics[k] is None:
            continue
            
        label_suffix = labels[k] if k < len(labels) else f"Exp_{k}"
        
        # 1. Plot Test Benchmarks (Horizontal Lines)
        try:
            res_path = os.path.join(path, 'results.csv')
            if os.path.exists(res_path):
                partial_df = pd.read_csv(res_path)
                test_r2 = partial_df['test_r2'].values[0]
                test_mse = partial_df['test_mse'].values[0]

                # Lines
                axes[1].axhline(y=test_r2, color='k', linestyle=':', label=f'Test {label_suffix}')
                axes[0].axhline(y=test_mse, color='k', linestyle=':', label=f'Test {label_suffix}')
                
                # Annotations
                # transform=... ensures 0.8 is 80% of the axis width, not data value 0.8
                axes[0].text(0.8, test_mse + 0.06, '{:.3f}'.format(test_mse), 
                             color='k', transform=axes[0].get_yaxis_transform(), fontsize=8)
                axes[1].text(0.8, test_r2 + 0.02, '{:.3f}'.format(test_r2), 
                             color='k', transform=axes[1].get_yaxis_transform(), fontsize=8)
        except Exception as e:
            print(f"Warning: Could not read results.csv for {path}: {e}")

        # 2. Handle Validation Metric Name
        # Ensure we have a 'loss' column to plot
        curr_val = val_metrics[k].copy()
        if 'loss' not in curr_val.columns and 'rmse' in curr_val.columns:
            curr_val['loss'] = curr_val['rmse']

        # 3. Plot Curves
        color = colormap[k % len(colormap)]
        
        # Loss (Top)
        axes[0].plot(train_metrics[k]['step'].values, train_metrics[k]['loss'].values, 
                     label=f'Train {label_suffix}', color=train_color, linestyle='--', alpha=0.5)
        axes[0].plot(curr_val['step'].values, curr_val['loss'].values, 
                     label=f'Validation {label_suffix}', color=color)
        
        # R2 (Bottom)
        axes[1].plot(train_metrics[k]['step'].values, train_metrics[k]['rsquare'].values, 
                     label=f'Train {label_suffix}', color=train_color, linestyle='--', alpha=0.5)
        axes[1].plot(curr_val['step'].values, curr_val['rsquare'].values, 
                     label=f'Validation {label_suffix}', color=color)

    # --- Formatting ---
    
    # Legend (Top Plot)
    axes[0].legend(bbox_to_anchor=(1.02, 1.3), ncol=3)
    
    # Y-Axis Settings
    axes[0].set_ylabel('Loss')
    axes[0].set_ylim(0., 1)
    
    axes[1].set_ylabel('R-square')
    axes[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    axes[1].set_ylim(0., 1)
    
    # X-Axis Settings (Log Scale)
    axes[0].set_xscale('log')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Iteration')
    
    return fig, axes
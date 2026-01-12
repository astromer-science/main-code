import matplotlib.pyplot as plt
import numpy as np
import os


def plot_gamma_weights(results, fig=None, axes=None):
    """
    Plots the gamma weights statistics for Alcock and ATLAS datasets across different SPC values.
    
    If 'fig' or 'axes' are None, a new 3x2 subplot grid is created with default styling.

    Args:
        results (dict): A dictionary containing the experiment results. 
                        Structure: results[ds_name][spc] = list_of_arrays
        fig (matplotlib.figure.Figure, optional): An existing figure object. Defaults to None.
        axes (numpy.ndarray, optional): An existing 3x2 array of axes. Defaults to None.

    Returns:
        tuple: A tuple containing (fig, axes).
    """
    
    # --- Default Initialization ---
    # If either fig or axes is missing, we create a new standard layout
    if fig is None or axes is None:
        fig, axes = plt.subplots(3, 2, figsize=(5, 5), dpi=100, sharex='col', sharey='row', 
                                 gridspec_kw={'hspace': 0.05, 'wspace': 0.05})

    # --- Constants & Config ---
    x_labels = [r'$\gamma_0$', r'$\gamma_1$', r'$\gamma_2$', r'$\gamma_3$', r'$\gamma_4$', r'$\gamma_5$', r'$\gamma_6$']
    ds_keys = ['alcock', 'atlas']
    ds_names = ['Alcock', 'ATLAS']
    spc_list = [20, 100, 500]
    
    limit_map = {
        20: (0.12, 0.165),
        100: (0.12, 0.175),
        500: (0.10, 0.185)
    }

    # --- Plotting Logic ---
    for i, dsname in enumerate(ds_keys):
        for k, spc in enumerate(spc_list):
            
            # Safety check for axes shape
            if k >= axes.shape[0] or i >= axes.shape[1]:
                continue

            # Data extraction
            data = results[dsname][spc]
            mean_values = np.mean(data, axis=0)
            std_values = np.std(data, axis=0)
            
            ax = axes[k][i]
            
            # Bar chart
            ax.bar(range(len(x_labels)), mean_values, yerr=std_values, 
                   color='#E1D5E7', ecolor='k', edgecolor='#9673A6')
            
            # Axis formatting
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels)

            # Value annotations
            for j, v in enumerate(mean_values):
                ax.text(j - 0.45, v + 0.004, '{:.2f}'.format(v), fontsize=6, color='black')

            # Y-Axis limits and ticks
            minv, supv = limit_map.get(spc, (0.1, 0.2))
            ax.set_ylim(minv, supv)
            ax.set_yticks([])

            # Side label (SPC) - only on the last column
            if i == len(ds_keys) - 1:
                ax_twin = ax.twinx()
                ax_twin.text(7.4, 0.2, '{} SPC'.format(spc), ha='right', fontsize=8, rotation=270)  
                ax_twin.set_yticks([])

        # Column titles - only on the first row
        axes[0][i].set_title(ds_names[i], fontsize=8)
    
    # Global Y label
    axes[1][0].set_ylabel('Weight Value', fontsize=8)
    
    return fig, axes
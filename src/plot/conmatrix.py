import matplotlib.pyplot as plt
import seaborn as sns

def _init_cm_layout(dataname):
    """
    Initializes the specific 2x2 layout with centered bottom plot based on dataset.
    """
    if dataname.lower() == 'alcock':
        figsize = (7, 7)
        gridspec = {'hspace': -0.2, 'wspace': 0.04}
    else: # ATLAS
        figsize = (6, 5.5)
        gridspec = {'hspace': -0.1, 'wspace': 0.04}
        
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True, gridspec_kw=gridspec)
    
    # Layout adjustments defined in original code
    fig.delaxes(axes[1, 1])
    axes[1, 0].set_position([0.25, 0.1, 0.5, 0.3])  # [left, bottom, width, height]
    
    # Flatten and keep relevant axes (0, 1, 2)
    axes_flat = axes.flatten()
    # We return the relevant axes for plotting (0, 1, and the moved 2)
    # Note: axes_flat[3] is the deleted one
    return fig, [axes_flat[0], axes_flat[1], axes_flat[2]]

def plot_confusion_matrices(cm_data, dataname, fig=None, axes=None):
    """
    Plots the confusion matrices for the given dataset statistics.
    
    Args:
        cm_data (dict): Output from compute_cm_stats.
        dataname (str): 'ATLAS' or 'Alcock'.
        fig (Figure, optional): Matplotlib figure.
        axes (list, optional): List of 3 axes.
    """
    
    # --- Configuration ---
    atlas_labels = ['CB', 'DB', 'Mira', 'Other', 'Pulse']
    alcock_labels = ['Cep_0', 'Cep_1', 'EC', 'LPV', 'RRab', 'RRc']
    
    labels_ = alcock_labels if dataname.lower() == 'alcock' else atlas_labels
    spc_list = [20, 100, 500]

    # --- Initialization ---
    if fig is None or axes is None:
        fig, axes = _init_cm_layout(dataname)

    # --- Plotting Loop ---
    for k, spc in enumerate(spc_list):
        if spc not in cm_data:
            continue
            
        mean_cm = cm_data[spc]['mean']
        std_cm = cm_data[spc]['std']
        ax = axes[k]

        # Heatmap
        sns.heatmap(mean_cm, annot=False, fmt='d', cmap='Purples',
                    xticklabels=labels_, yticklabels=labels_, ax=ax, cbar=False)
        
        # Custom Annotations
        for i in range(mean_cm.shape[0]):
            for j in range(mean_cm.shape[1]):
                mean_val = mean_cm[i, j]
                std_val = std_cm[i, j] 
                
                text = r''
                if mean_val >= 0.01:
                    text = "{:.2f}\n$\pm${:.2f}".format(mean_val, std_val)
                
                color = 'w' if i == j else 'k'
                
                ax.text(j + 0.5, i + 0.5, text,
                        ha='center', va='center', color=color, fontsize=8.5)

        ax.set_title('{} samples per class'.format(spc))

    # --- Global Labels ---
    # Assuming axes[2] is the bottom one and axes[0] is top-left
    axes[2].set_xlabel('Predicted label')
    axes[0].set_ylabel('True label')
    
    return fig, axes
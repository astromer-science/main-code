import seaborn as sns
import numpy as np

def plot_cm(cm, ax, title='CM', fontsize=15, cbar=False, yticklabels=True, class_names=None):
    '''
    Plot Confusion Matrix
    '''
    labels = np.zeros_like(cm, dtype=np.object)
    mask = np.ones_like(cm, dtype=np.bool)
    for (row, col), value in np.ndenumerate(cm):
        if value != 0.0:
            mask[row][col] = False
        if value < 0.01:
            labels[row][col] = '< 1%'
        else:
            labels[row][col] = '{:2.1f}%'.format(value*100)

    ax = sns.heatmap(cm, annot = labels, fmt = '',
                     annot_kws={"size": fontsize},
                     cbar=cbar,
                     ax=ax,
                     linecolor='white',
                     linewidths=1,
                     vmin=0, vmax=1,
                     cmap='Blues',
                     mask=mask,
                     yticklabels=yticklabels)

    try:
        if yticklabels and class_names is not None:
            ax.set_yticklabels(class_names, rotation=0, fontsize=fontsize+1)
            ax.set_xticklabels(class_names, rotation=90, fontsize=fontsize+1)
    except:
        pass
    ax.set_title(title, fontsize=fontsize+5)

    ax.axhline(y=0, color='k',linewidth=4)
    ax.axhline(y=cm.shape[1], color='k',linewidth=4)
    ax.axvline(x=0, color='k',linewidth=4)
    ax.axvline(x=cm.shape[0], color='k',linewidth=4)

    return ax

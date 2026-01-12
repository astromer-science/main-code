import os

def save_plot(fig, path):
    """
    Saves the provided figure to the specified path, creating directories if necessary.

    Args:
        fig (matplotlib.figure.Figure): The figure object to be saved.
        path (str): The full file path including extension (e.g., './plots/figure.pdf').
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    fig.savefig(path, bbox_inches='tight')
    print(f"Plot saved successfully at: {path}")
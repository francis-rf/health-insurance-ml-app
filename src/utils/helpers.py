
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df, out_path=None):
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap='vlag', center=0)
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()

def save_figures(fig, path):
    fig.savefig(path, bbox_inches='tight')

def load_processed_data(path):
    return pd.read_csv(path)

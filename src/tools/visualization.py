# Functions to build visuals

# Libraries
import seaborn as sns
from matplotlib import pyplot as plt

# --------------------------------------------------------------------------------------------------------------------
# Function to draw barplots of model's RMSE:
def plot_model_RMSE(df, title):
    fig = sns.barplot(data=df, x="RMSE", y="Model", hue ='Condition')
    fig.set(xlim=(0, 0.12))
    fig.bar_label(fig.containers[0], padding = 2)
    fig.bar_label(fig.containers[1], padding = 2)
    plt.title(title, weight='bold', y = 1.02)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    return fig
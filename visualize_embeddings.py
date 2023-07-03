import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json

def visualize(df, name):
    labels = df['word']
    matrix = np.array(df['ada_embedding'].apply(json.loads).to_list())

    # Create a t-SNE model and transform the data
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(matrix) 

    x = [x for x,y in vis_dims]
    y = [y for x,y in vis_dims]

    fig, ax = plt.subplots(figsize=(10,8))

    ax.scatter(x, y)
    for i in range(len(x)):
        ax.annotate(labels[i], (x[i], y[i]))
    ax.set_title(f"{name.capitalize()} visualized using t-SNE")
    plt.savefig(f'{name}.png')

birds_df = pd.read_csv('embedded_birds.csv')
visualize(birds_df, 'birds')

soups_df = pd.read_csv('embedded_soups.csv')
visualize(soups_df, 'soups')
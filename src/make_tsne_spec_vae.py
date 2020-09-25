"""
Makes a TSNE plot for the embeddings from the spectrogram-VAE color-coded by artists
"""

import time
import pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

# path to the pickle file with spectrogram-VAE embeddings
PKL_PATH = 'data/processed/nlp4musa/image_mu.pkl'


def make_tsne():
    # load pkl file
    with open(PKL_PATH, 'rb') as f:
        # key-value pairs
        artist_embeddings_dict = pickle.load(f)

    artists = []
    spec_embeddings = []
    for k, v in artist_embeddings_dict.items():
        # add artists here to preserve order
        artists.append(k.split('_')[0].lower())  # k actually has album id too
        spec_embeddings.append(v[0])

    spec_embeddings = np.array(spec_embeddings)
    artists = artists
    print(f'Spec embeddings dimensions: {spec_embeddings.shape}')

    # reduce embeddings to 70 dimensions using PCA to easen the task
    # of t-SNE algorithm later on
    pca = PCA(n_components=70)
    # reduced spectrogram embeddigns
    spec_embeddings = pca.fit_transform(spec_embeddings)
    print(f'Reduced spec dimensions: {spec_embeddings.shape}')
    print('Cumulative explained variation for 70 principal components:')
    print(np.sum(pca.explained_variance_ratio_))

    # t-SNE part
    # n_components=2 since we need to plot a 2-D graph
    tsne = TSNE(n_components=2)
    print('Starting t-SNE transformation...')
    start = time.time()
    spec_embeddings = tsne.fit_transform(spec_embeddings)
    print(f't-SNE done! [{time.time()-start}]')
    print(f't-SNE dimensions: {spec_embeddings.shape}')

    # construct a dataframe for visualization
    # TODO: remove 10000 limit from final version
    viz_df = pd.DataFrame()
    viz_df['tsne-dim-0'] = spec_embeddings[:, 0]
    viz_df['tsne-dim-1'] = spec_embeddings[:, 1]
    viz_df['artist'] = artists

    # plot
    uniq_artists = set(artists)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x='tsne-dim-0',
                    y='tsne-dim-1',
                    hue='artist',
                    palette=sns.color_palette('hls', len(uniq_artists)),
                    legend='full',
                    data=viz_df)
    plt.savefig('tsne.png')


if __name__ == '__main__':
    make_tsne()

from pathlib import Path

import numpy as np
import pandas as pd
import umap

DEFAULT = np.random.randn(100)
MISSING_EMB = lambda x: DEFAULT
# MISSING_EMB = np.random.randn


def find_embedding_by_item_id(item_id, embeddings):
    emb_len = len(embeddings.iloc[0]['embedding'].split())

    if not item_id.isdigit():
        return MISSING_EMB(emb_len)

    embedding = embeddings[embeddings['item_id'] == int(item_id)]

    if len(embedding) > 1:
        assert f"Duplicates {embedding}"
    elif len(embedding) == 0:
        # print("Item_id missing ", item_id)
        return MISSING_EMB(emb_len)

    emb_vector = embedding['embedding'].values[0].split()
    emb_vector = np.array(list(map(float, emb_vector)))
    if np.all(emb_vector == 0):
        return MISSING_EMB(emb_len)
    return emb_vector


def load_embedding(dataset):
    embedding_file_path = Path(dataset.dataset_path) / f'{dataset.dataset_name}.item'
    embeddings_df = pd.read_csv(embedding_file_path, sep='\t', skiprows=[0], names=['item_id', 'embedding'])

    item_tokens_id = dataset.field2id_token['item_id']

    embeddings = [find_embedding_by_item_id(token_id, embeddings_df) for token_id in item_tokens_id]
    return np.array(embeddings)


def resize_embedding(embeddings, dim):
    reducer = umap.UMAP(n_components=dim, random_state=2023)
    umap_emb = reducer.fit_transform(embeddings)
    return umap_emb



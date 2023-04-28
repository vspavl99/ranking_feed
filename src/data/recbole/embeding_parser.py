import ast
import random
from typing import List, Callable

import numpy as np
import pandas as pd

random.seed(2023)


def read_vec_file(path_to_file: str) -> List[tuple]:
    """
    Read .vec file line by line.
    :param path_to_file: path to file .vec
    :return: list of (id, emd_vector)
    """
    result = []
    with open(path_to_file, 'r') as vec_file:
        while True:
            id_and_vec_line = vec_file.readline()

            if not id_and_vec_line:
                break

            splitted_data = id_and_vec_line.split()
            _id, vector = splitted_data[0], splitted_data[1:]
            result.append((_id, np.asarray(vector, dtype=float)))

    return result


def match_news_and_embedding(
        news_dataframe: pd.DataFrame, embeddings: pd.DataFrame,
        entity_name: str = 'title_entities', agg_func: Callable = random.choice, emb_as_string = True):
    """
    Create Dataframe with new_id and corresponding entity_id. (Entities specified by agg_func and entity_name)
    :param news_dataframe: dataframe with news additional data
    :param embeddings: dataframe with embeddings
    :param entity_name: title_entities or abstract_entities
    :param agg_func: determine how choose entities if several available
    :return:
    """

    news_x_embedding = []
    for i, news_detailed_info in news_dataframe.iterrows():

        entities = ast.literal_eval(news_detailed_info[entity_name])
        embedding = np.zeros(100)

        if len(entities):
            entity = agg_func(entities)
            try:
                embedding = embeddings[embeddings['entity_id'] == entity['WikidataId']]['embedding'].values[0]
            except Exception as e:
                print(entity['WikidataId'])

        if emb_as_string:
            embedding = " ".join(map(str, embedding))
        news_x_embedding.append((news_detailed_info['news_id'].strip('N'), embedding))

    # news_x_embedding = pd.DataFrame(news_x_embedding, columns=['item_id:token', 'embedding:float_seq'])
    return news_x_embedding


def save_emb_file(data, path):
    with open(path, 'w') as file:
        file.write("item_id:token\tembedding:float_seq\n")
        for row in data:
            file.write(f"{row[0]}\t{row[1]}\n")


def save_emb_file_splitted_per_column(data, path):
    with open(path, 'w') as file:

        file.write("item_id:token\t" + "\t".join([f"{i}:float" for i in range(len(list(map(float, data[0][1].split()))))]) + "\n")
        for row in data:
            file.write(f"{row[0]}\t" + "\t".join(data[0][1].split()) + "\n")


def main(dataset_name='MINDsmall'):
    all_item_emb = []
    for dataset_split in ['dev', 'train']:

        new_dataset_name = dataset_name[:4].lower() + '_' + dataset_name[4:]
        embeddings = read_vec_file(f'data/raw/MIND_dataset/{dataset_name}_{dataset_split}/entity_embedding.vec')
        embeddings = pd.DataFrame(embeddings, columns=['entity_id', 'embedding'])
        embeddings.to_csv(f'data/processed/MIND_dataset/{new_dataset_name}/entity_embedding.csv')

        news_info = pd.read_csv(
            filepath_or_buffer=f'data/raw/MIND_dataset/{dataset_name}_{dataset_split}/news.tsv',
            names=['news_id', 'category', 'sub_category', 'title',
                   'abstract', 'url', 'title_entities', 'abstract_entities'],
            sep=r'\t',
            engine='python'
        )

        # entity_name and agg_func can be specified in match_news_and_entities
        news_x_embedding = match_news_and_embedding(news_info, embeddings, entity_name='title_entities')

        all_item_emb.extend(news_x_embedding)

    save_emb_file(all_item_emb, path=f'data/processed/MIND_dataset/{new_dataset_name}/{new_dataset_name}.item')
    save_emb_file_splitted_per_column(all_item_emb, path=f'data/processed/MIND_dataset/{new_dataset_name}/{new_dataset_name}_columns.item')


if __name__ == '__main__':
    main()



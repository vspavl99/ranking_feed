import os
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd

DEFAULT_NAME = 'mind_experimental'
np.random.seed(2023)


def select_random_users(number_of_users: int, interactions: pd.DataFrame):
    all_users = interactions['user_id'].unique()
    return np.random.choice(all_users, number_of_users)


def write_file(filename, data):
    with open(filename, 'w') as file:
        for row in data.iterrows():
            file.write(row)


def save_new_dataset(save_dir: str, data: pd.DataFrame, filename):

    # Create separate dir for new dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data.to_csv(filename, sep='\t', index=False)


def create_debug_split(number_of_users: int = 2, data_dir: str = 'data/processed/MIND_dataset',
                       dataset_name: str = 'mind_small', split: str = 'train') -> NoReturn:
    """
    Extract from given dataset subset of random users (This split-dataset is using for fast debugging model/pipelines)
    :param number_of_users: number of users for extracting
    :param data_dir: path where dataset located
    :param dataset_name: name of dataset for extracting subset
    :param split: 'train' or 'dev'
    :return: Create new dataset in data_dir
    """

    path_to_interactions = Path(data_dir) / dataset_name / f'{dataset_name}.{split}.inter'
    user_item_interactions = pd.read_csv(path_to_interactions, sep='\t', skiprows=[0],
                            names=['user_id', 'item_id', 'label', 'timestamp'])

    users = select_random_users(number_of_users, user_item_interactions)
    subset_interactions = user_item_interactions[user_item_interactions['user_id'].isin(users)]
    print("Shape of extracted interactions: ", subset_interactions.shape)

    save_dir = os.path.join(data_dir, DEFAULT_NAME)
    subset_interactions_filepath = Path(data_dir) / DEFAULT_NAME / f'{DEFAULT_NAME}.{split}.inter'
    subset_interactions = subset_interactions.rename(
        columns={'user_id': 'user_id:token', 'item_id': 'item_id:token',
                 'label': 'label:float', 'timestamp': 'timestamp:float'}
    )
    save_new_dataset(save_dir, subset_interactions, subset_interactions_filepath)

    path_to_item_features = Path(data_dir) / dataset_name / f'{dataset_name}.item'
    item_features = pd.read_csv(path_to_item_features, sep='\t', skiprows=[0], names=['item_id', 'embedding'])

    subset_user_item_features = item_features[item_features['item_id'].isin(subset_interactions['item_id:token'])]
    subset_user_item_features_filepath = Path(data_dir) / DEFAULT_NAME / f'{DEFAULT_NAME}.{split}.item'
    subset_user_item_features = subset_user_item_features.rename(
        columns={'item_id': 'item_id:token', 'embedding': 'embedding:float_seq'}
    )
    print("Shape of extracted items: ", subset_user_item_features.shape)
    save_new_dataset(save_dir, subset_user_item_features, subset_user_item_features_filepath)


if __name__ == '__main__':
    create_debug_split(split='train')

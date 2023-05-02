from src.data.recbole.embeding_parser import create_item_data
from src.data.recbole.prepare_mind_to_recbole import create_recbole_dataset
from src.data.recbole.combine_item_features import combine_features


def full_pipeline():
    """Full pipeline of data preparation for training"""

    create_recbole_dataset(
        dataset='mind_small_dev',
        input_path='data/raw/MIND_dataset/MINDsmall_dev',
        output_path='data/processed/MIND_dataset/mind_small'
    )

    create_item_data(dataset_name='MINDsmall', subsets=['train', 'dev'])
    combine_features(data_dir='data/processed/MIND_dataset', dataset_name='mind_small')


if __name__ == '__main__':
    full_pipeline()

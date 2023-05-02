import pandas as pd


def combine_features(data_dir: str = 'data/processed/MIND_dataset/', dataset_name: str = 'mind_experimental'):
    path = f'{data_dir}/{dataset_name}/{dataset_name}.train.item'
    train_feat = pd.read_csv(path, sep='\t')
    print(train_feat.shape)

    path = f'{data_dir}/{dataset_name}/{dataset_name}.dev.item'
    dev_feat = pd.read_csv(path, sep='\t')
    print(dev_feat.shape)

    features = pd.concat((train_feat, dev_feat))
    print(features.shape)

    features = features.drop_duplicates()
    print(features.shape)

    path = f'{data_dir}/{dataset_name}/{dataset_name}.item'
    features.to_csv(path, sep='\t', index=False)


if __name__ == '__main__':
    combine_features()

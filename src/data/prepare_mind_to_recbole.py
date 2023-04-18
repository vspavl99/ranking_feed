import click
import importlib
from src.data.RecDatasets.conversion_tools.src.utils import dataset2class


@click.command()
@click.option('--dataset', type=str, help='dataset name (mind_large_train/mind_large_dev, etc)')
@click.option('--input_path', type=str, help='Path to a folder where dataset located')
@click.option('--output_path', type=str, help='Path to a folder where to save prepared data.')
def main(dataset, input_path, output_path):

    input_args = [input_path, output_path]
    dataset_class_name = dataset2class[dataset.lower()]
    dataset_class = getattr(
        importlib.import_module('src.data.RecDatasets.conversion_tools.src.extended_dataset'), dataset_class_name
    )

    datasets = dataset_class(*input_args)
    datasets.convert_inter()


if __name__ == '__main__':
    main()

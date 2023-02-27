import os.path
import zipfile

import click

from src.data.common import create_baked_data_folders


@click.command()
@click.option('--raw_data_dir', type=str, help='Path to a folder with raw TTRS data.')
@click.option('--baked_data_dir', type=str, help='Path to a folder where to save prepared TTRS data.')
def main(raw_data_dir: str, baked_data_dir: str):

    create_baked_data_folders(baked_data_dir)
    unzip_archives(raw_data_dir, baked_data_dir)


def unzip_archives(raw_data_dir: str, baked_data_dir: str):
    raw_data_zip = os.path.join(raw_data_dir, 'TTRS_dataset.zip')
    with zipfile.ZipFile(raw_data_zip, 'r') as zip_ref:

        zip_ref.extractall(baked_data_dir)


if __name__ == '__main__':
    main()

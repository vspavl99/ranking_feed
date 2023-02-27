import os.path
import zipfile

import click

from src.data.common import create_baked_data_folders


@click.command()
@click.option('--raw_data_dir', type=str, help='Path to a folder with raw MIND data.')
@click.option('--baked_data_dir', type=str, help='Path to a folder where to save prepared MIND data.')
def main(raw_data_dir: str, baked_data_dir: str):

    create_baked_data_folders(baked_data_dir)
    unzip_archives(raw_data_dir, baked_data_dir)


def unzip_archives(raw_data_dir: str, baked_data_dir: str):
    zip_files = ['MINDlarge_dev.zip', 'MINDlarge_train.zip']

    for zip_file in zip_files:
        raw_data_zip = os.path.join(raw_data_dir, zip_file)
        with zipfile.ZipFile(raw_data_zip, 'r') as zip_ref:

            out_path = os.path.join(baked_data_dir, zip_file.split('.zip')[0])
            os.makedirs(out_path)

            zip_ref.extractall(out_path)


if __name__ == '__main__':
    main()

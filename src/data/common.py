import os.path
import zipfile

import click


def create_baked_data_folders(baked_data_dir: str):
    os.makedirs(baked_data_dir, exist_ok=True)
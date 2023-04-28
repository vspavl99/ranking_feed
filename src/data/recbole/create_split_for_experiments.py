import os
from pathlib import Path
import pandas as pd

DEFAULT_NAME = 'mind_experimental'


def read_first_n_row(filename, n):
    data = []
    with open(filename, 'r') as file:
        for _ in range(n):
            data.append(file.readline())
    return data


def write_file(filename, data):
    with open(filename, 'w') as file:
        for row in data:
            file.write(row)


def main(number_of_rows=100, data_dir='data/processed/MIND_dataset',
         base_dataset_name='mind_small'):

    save_dir = os.path.join(data_dir, DEFAULT_NAME)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    item = read_first_n_row(Path(data_dir) / base_dataset_name / f'{base_dataset_name}.item', number_of_rows)
    write_file(os.path.join(save_dir, f'{DEFAULT_NAME}.item'), item)

    inter = read_first_n_row(Path(data_dir) / base_dataset_name / f'{base_dataset_name}.train.inter', number_of_rows)
    write_file(os.path.join(save_dir, f'{DEFAULT_NAME}.train.inter'), inter)


if __name__ == '__main__':
    main()

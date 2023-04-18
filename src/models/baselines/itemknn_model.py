from recbole.quick_start import run_recbole

if __name__ == '__main__':
    run_recbole(
        model='ItemKNN',
        dataset='mind_large',
        config_file_list=['config/experiments/ItemKNN_model.yaml']
    )

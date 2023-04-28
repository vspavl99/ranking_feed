from recbole.quick_start import run_recbole

if __name__ == '__main__':
    run_recbole(
        model='Pop',
        dataset='mind_large',
        config_file_list=['config/experiments_baselines/pop_model.yaml']
    )
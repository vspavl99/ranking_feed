from recbole.quick_start import run_recbole

if __name__ == '__main__':
    run_recbole(
        model='DMF',
        dataset='mind_small',
        config_file_list=['config/experiments_baselines/DMF_model.yaml']
    )

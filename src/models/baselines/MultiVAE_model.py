from recbole.quick_start import run_recbole

if __name__ == '__main__':
    parameter_dict = {
        'train_neg_sample_args': None,
    }

    run_recbole(
        model='MultiVAE',
        dataset='mind_small',
        config_file_list=['config/experiments_baselines/MultiVAE_model.yaml'],
        config_dict=parameter_dict
    )

from recbole.quick_start import run_recbole

if __name__ == '__main__':
    model = 'LR'

    run_recbole(
        model=model,
        dataset='mind_small',
        config_file_list=[f'config/experiments_external_data/{model}.yaml'],
        config_dict={'log_wandb': False}
    )

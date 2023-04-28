from recbole.quick_start import run_recbole

if __name__ == '__main__':

    run_recbole(
        model='DSSM',
        dataset='mind_experimental',
        config_file_list=['config/experiments_external_data/DSSM_test.yaml']
    )

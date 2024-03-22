from finetuning import start_training


class config:
    input_dim = 1024
    hidden_dim_d = 128
    hidden_dim_t = 128
    hidden_dim_c = 128
    lr = 1e-4
    dr = 0.1
    patience = 15
    accelerator = 'gpu'
    max_epochs = 100
    dataset_name = '../analysis/kiba_cliff_pairs_ta1_ts0.9_r_wt.csv'


start_training(mode='DDC', config=config, project_name='DDC_3_')


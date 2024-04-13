from finetuning import start_training


class config:
    n_hidden_layers = 1
    hidden_dim_d = 1024
    hidden_dim_t = 756
    hidden_dim_c = 756
    lr = 0.0003
    dr = 0.01
    patience = 15
    accelerator = 'gpu'
    max_epochs = 100
    dataset_name = './analysis/kiba_cliff_pairs_ta1_ts0.9_r_wt.csv'


start_training(mode='DDC', config=config, project_name='DDC_r_best')


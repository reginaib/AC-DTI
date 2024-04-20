from finetuning import start_training


class config:

    n_hidden_layers = 1
    n_targets = 676
    hidden_dim_d = 1024
    hidden_dim_t = 756
    hidden_dim_c = 756
    lr = 0.0003
    dr = 0.01
    patience = 15
    accelerator = 'gpu'
    max_epochs = 100
    dataset_name = '../analysis/bindingdb_ki_cliff_pairs_ta1_ts0.9_r_wt.csv'
    save_preds = False


start_training(mode='DDC', config=config, project_name='DDC_r_best')


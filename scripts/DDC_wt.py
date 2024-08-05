from finetuning import start_training


class config:
    n_hidden_layers = 1
    n_targets = 666
    hidden_dim_d = 256
    hidden_dim_t = 128
    hidden_dim_c = 1024
    lr = 0.001
    dr = 0.2
    patience = 15
    accelerator = 'gpu'
    max_epochs = 5
    csv = './analysis/kiba_pairs_aff_diffs_split_r_wt.csv'
    save_preds = False
    checkpoint = False
    task = 'regression'


start_training(mode='DDC', config=config, project_name='DDC_ACNet_tryout')


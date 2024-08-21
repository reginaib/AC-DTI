from finetuning import start_training


class config:
    n_hidden_layers = 2
    n_targets = 229
    hidden_dim_d = 128
    hidden_dim_t = 256
    hidden_dim_c = 768
    lr = 0.001
    dr = 0.2
    patience = 15
    accelerator = 'gpu'
    max_epochs = 100
    csv = './analysis/kiba_ddc_cb_ta1_ts0.9.csv'
    save_preds = False
    checkpoint = True


start_training(mode='DDC', config=config, project_name='DDC_KIBA_cb_best_train')


from finetuning import start_training


class config:
    n_hidden_layers = 1
    n_additional_dense_layers = 3
    hidden_dim_d = 1024
    hidden_dim_t = 256
    hidden_dim_c = 1024
    lr = 0.0001
    dr = 0.2
    patience = 15
    accelerator = 'gpu'
    max_epochs = 100
    dataset_name = './analysis/kiba_d_t_aff_smiles_split.csv'
    #pre_trained_d_encoder_path = './results/DTI_epoch_99.ckpt'
    pre_trained_d_encoder_path = None
    preds_name = 'DTI_tl_adl_3_preds'
    freeze = False
    layer_to_d_encoder = False
    hidden_dim_d_add = 32
    dr2 = 0.1
start_training(mode='DTI', config=config, project_name='DTI_tl_best')

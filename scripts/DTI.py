from finetuning import start_training


class config:
    n_hidden_layers = 1
    hidden_dim_d = 1024
    hidden_dim_t = 768
    hidden_dim_c = 1024
    lr = 0.0001
    dr = 0.01
    patience = 15
    accelerator = 'gpu'
    max_epochs = 100
    dataset_name = './analysis/kiba_d_t_aff_smiles_split.csv'
    #pre_trained_d_encoder_path = './results/DDC_best_epoch_23.ckpt'
    pre_trained_d_encoder_path = None
    preds_name = 'DTI_tl_adl_3_preds'
    freeze = True
    layer_to_d_encoder = True
    hidden_dim_d_add = 1024
    dr2 = 0.5
    save_preds = False

start_training(mode='DTI', config=config, project_name='DTI_tl_add_l_to_d_enc_best')

from finetuning import start_sweep


sweep_config = {
        "method": "random",
        "metric": {
            "goal": "minimize",
            "name": "Validation/MAELoss"
        },
        "parameters": {
            "mode": {"value": "DTI"},
            "n_targets": {"value": 676},
            "n_hidden_layers": {"values": [1, 2, 3]},
            "hidden_dim_d": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            "hidden_dim_t": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            "hidden_dim_c": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            "lr": {"values": [0.00001, 0.00003, 0.0001, 0.0003, 0.001]},
            "dr": {'values': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]},
            "patience": {"value": 15},
            "accelerator": {"value": "gpu"},
            "max_epochs": {"value": 100},
            "dataset_name": {"value": './analysis/kiba_d_t_aff_smiles_split.csv'},
            "pre_trained_d_encoder_path": {"value": None},
            #"pre_trained_d_encoder_path": {"value": './results/DDC_best_epoch_23.ckpt'},
            #"freeze": {"values": [True, False]},
            #"layer_to_d_encoder": {"values": [True, False]},
            #"hidden_dim_d_add": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            #"dr2": {'values': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]},
            "save_preds": {"value": False},
        }
}


start_sweep(config=sweep_config, project_name='DTI_sweep_vary_all', num_config=100)

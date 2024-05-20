from finetuning import start_sweep


sweep_config = {
        "method": "random",
        "metric": {
            "goal": "minimize",
            "name": "Validation/MAELoss"
        },
        "parameters": {
            "mode": {"value": "DTI"},
            "n_targets": {"value": 1018},
            "n_hidden_layers": {"values": [1, 2, 3, 4]},
            "hidden_dim_d": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            "hidden_dim_t": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            "hidden_dim_c": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            "lr": {"values": [0.00001, 0.00003, 0.0001, 0.0003, 0.001]},
            "dr": {'values': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]},
            "patience": {"value": 15},
            "accelerator": {"value": "gpu"},
            "max_epochs": {"value": 100},
            "dataset_name": {"value": './analysis/bindingdb_ki_d_t_aff_smiles_split-2.csv'},
            "pre_trained_d_encoder_path": {"value": None},
            # "pre_trained_d_encoder_path": {"value": './results/DDC_BDB_best_epoch_23.ckpt'},
            "freeze": {"value": False},
            "layer_to_d_encoder": {"value": False},
            "hidden_dim_d_add": {"value": 0},
            "dr2": {'value': 0},
            "save_preds": {"value": False},
            "checkpoint": {"value": False},

        }
}


start_sweep(config=sweep_config, project_name='DTI_BDB_sweep_vary_all', num_config=100)

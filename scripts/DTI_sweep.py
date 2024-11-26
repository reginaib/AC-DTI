from finetuning import start_sweep


# Define the configuration for the hyperparameter sweep
sweep_config = {
        "method": "random",
        "metric": {
            "goal": "minimize",
            "name": "Validation/MAELoss"
        },
        "parameters": {
            "mode": {"value": "DTI"},
            "n_targets": {"value": 229},
            # "n_hidden_layers": {"value": 1},
            "n_hidden_layers": {"values": [1, 2, 3, 4]},
            # "hidden_dim_d": {"value": 512},
            "hidden_dim_d": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            "hidden_dim_t": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            "hidden_dim_c": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            "lr": {"values": [0.00001, 0.00003, 0.0001, 0.0003, 0.001]},
            "dr": {'values': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]},
            "patience": {"value": 15},
            "accelerator": {"value": "gpu"},
            "max_epochs": {"value": 100},
            "dataset_name": {"value": '../analysis/kiba_dti_cb_split.csv'},
            "pre_trained_d_encoder_path": {"value": None},
            #"pre_trained_d_encoder_path": {"value": '../results/DDC_KIBA_sim_aff_diff_best_epoch_96.ckpt'},
            "pre_trained_t_encoder_path": {"value": None},
            #"pre_trained_t_encoder_path": {"value": '../results/DDC_KIBA_sim_aff_diff_best_epoch_96.ckpt'},
            "freeze": {"value": False},
            "layer_to_d_encoder": {"value": False},
            "hidden_dim_d_add": {"value": 0},
            "dr2": {'value': 0},
            "save_preds": {"value": False},
            "checkpoint": {"value": False},
        }
}

# Start the hyperparameter sweep with the defined configuration
# Arguments:
# - config: The configuration dictionary for the sweep
# - project_name: The name of the project in Weights and Biases where the sweep results will be logged
# - num_config: The number of different configurations to test during the sweep
start_sweep(config=sweep_config, project_name='DTI_KIBA_cb_sweep', num_config=100)

`cliffs.py`: use  `get_similarity_matrix` method for comparing chemical structures represented by SMILES strings. 
It evaluates similarity based on different criteria including Levenshtein distance, structural similarity, and scaffold similarity.

`name-of-dataset.ipynb` files contain postprocessing steps for activity cliffs identification for the relevant dataset

## Performance evaluation

1. Making predictions. 

Prerequisites:
Ensure that DeepPurpose is installed in your environment as `predict_model-name_dataset.ipynb` relies on DeepPurpose methods.

- Go to `predict_model-name_dataset.ipynb`, load the dataset and the (pretrained) model. 

Note: 'morgan_cnn_kiba' model matches to 'model_morgan_aac_kiba' when loading (specified in `utils.py` in DeepPurpose)

- Get predictions and assess performance by following the steps in the script. 
- Output files. Save the model predictions in `../analysis/model-name_dataset_predictions.csv`. Record the performance metrics in the first column of  `../analysis/model-name_dataset_performance.csv`.

2. Postprocessing. 
- Go to `dataset.ipynb` file, load targets and drugs of the dataset.
- Import the predictions you generated earlier to `unpivoted`.
- Follow the steps in the script and assess the performance. 
- Output files. Append performance metrics from the postprocessing to the `../analysis/model-name_dataset_performance.csv` file.

Note: The performance is calculated individually for both the first and second elements in each activity cliff pair. An overall performance calculation across all pairs is also feasible.
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
from cliffs import get_similarity_matrix
from tqdm import tqdm
import torch
from rdkit import Chem


def get_pairs(data, threshold_affinity=1, threshold_similarity=0.9):
    """
    Generates pairs of drugs according to the given thresholds.

    Args:
        data (DataFrame): The input data containing drug-target interactions.
        threshold_affinity (float, optional): The minimum difference in affinity
                                              between two drugs to consider them a significant pair. Defaults to 1.
        threshold_similarity (float, optional): The similarity threshold for drug pairs.
                                                Only pairs above this threshold are considered. Defaults to 0.9.

    Returns:
        DataFrame: A DataFrame containing paired drugs according to the given thresholds.
    """
    
    groups = []

    # Loop through each group in the DataFrame grouped by 'target'
    # 'g_name' holds the name of the target, 'group' contains the corresponding rows (related to specific target)
    for g_name, group in data.groupby('target', sort=False):
        # Calculate the similarity matrix for the drug molecules (related to specific target)
        sim = get_similarity_matrix(group.SMILES.to_list(), similarity=threshold_similarity)
        # Find non-zero elements in the similarity matrix, indicating pairs of similar drugs
        i, j = sim.nonzero()

        # Select corresponding rows from the affinity DataFrame for these drug pairs d1-d2
        # 'd1' and 'd2' represent the first and second drug in the pair respectively
        d1 = group.iloc[i]
        d2 = group.iloc[j]

        # Calculate the 1x difference in affinity between the two drugs (KIBA values)
        affinity_diff = np.abs(d1.affinity.values - d2.affinity.values) > threshold_affinity

        # Select rows from d1 and d2 where the affinity difference is significant
        cliff1 = d1[['drug', 'target', 'affinity', 'predicted']].iloc[affinity_diff]
        cliff2 = d2[['drug', 'target', 'affinity', 'predicted']].iloc[affinity_diff]

        # Pair up corresponding rows from cliff1 and cliff2 side by side
        paired = pd.concat([cliff1.reset_index(drop=True), cliff2.reset_index(drop=True)], axis=1)
        paired.columns = ['drug1', 'target', 'affinity1', 'predicted1', 'drug2', 'remove', 'affinity2', 'predicted2']
        paired = paired[['target', 'drug1', 'drug2', 'affinity1', 'affinity2', 'predicted1', 'predicted2']].copy()
        groups.append(paired)

    # Concatenate all group DataFrames into a single DataFrame
    groups = pd.concat(groups)
    return groups


def r2_rmse(data):
    """
    Computes the R2 and RMSE for a given group of predictions.

    Args:
        g (DataFrame): The input data containing true and predicted affinity values.

    Returns:
        Series: A pandas Series containing the R2 and RMSE values.
    """

    r2 = r2_score(data['affinity'], data['predicted'])
    rmse = np.sqrt(mean_squared_error(data['affinity'], data['predicted']))
    return pd.Series({'r2': r2, 'rmse': rmse})


def get_performance(data, metric, averaging):
    """
    Computes performance metrics (R2, RMSE) for drug pairs.

    Args:
        data (DataFrame): The input data containing drug pairs and their affinity values.
        metric (function): The metric function to apply (e.g., r2_rmse).
        averaging (str): Type of averaging to perform ('micro' or 'macro').

    Returns:
        Series or DataFrame: Performance metrics averaged either across all data ('micro')
                             or per target ('macro').
    """

    # Prepare the data by splitting it into two sets based on drug pairs
    df1 = data[['target', 'drug1', 'affinity1', 'predicted1']]
    df2 = data[['drug2', 'affinity2', 'predicted2']]

    # Concatenate the two sets back together after renaming columns for consistency
    new_df = pd.concat([df1.rename(columns={'drug1': 'drug', 'affinity1': 'affinity',
                                            'predicted1': 'predicted'}),
                        df2.rename(columns={'drug2': 'drug', 'affinity2': 'affinity',
                                            'predicted2': 'predicted'})], ignore_index=True)

    if averaging == 'micro':
        return metric(new_df)

    if averaging == 'macro':
        return new_df.groupby('target').apply(metric).mean()


def get_total_performance(data, metric, averaging):
    """
    Computes the overall performance metrics for the dataset.

    Args:
        data (DataFrame): The input data containing drug-target pairs.
        metric (function): The metric function to apply (e.g., r2_rmse).
        averaging (str): Type of averaging to perform ('micro' or 'macro').

    Returns:
        Series or DataFrame: Overall performance metrics across the dataset.
    """

    if averaging == 'micro':
        return metric(data)

    if averaging == 'macro':
        return data.groupby('target_id').apply(metric).mean()


def get_total_metrics(data, threshold_affinity: list[float], threshold_similarity: list[float]):
    """
    Computes performance metrics across different affinity and similarity thresholds.

    Args:
        data (DataFrame): The input data containing drug-target interactions.
        threshold_affinity (list[float]): A list of affinity thresholds to evaluate.
        threshold_similarity (list[float]): A list of similarity thresholds to evaluate.

    Returns:
        DataFrame: A pandas DataFrame containing the computed metrics for each combination
                   of affinity and similarity thresholds.
    """

    # Calculate the total number of iterations required for all threshold combinations
    total_iterations = len(threshold_affinity) * len(threshold_similarity)
    results = []

    # Initialize the progress bar for processing all threshold combinations
    with tqdm(total=total_iterations, desc="Processing All Thresholds") as pbar:
        for ta in threshold_affinity:
            for ts in threshold_similarity:
                # Generate pairs of drugs for the current combination of thresholds
                groups = get_pairs(data, ta, ts)

                # Skip if no pairs are found for the current thresholds
                if groups.empty:
                    print(f"No data available for threshold_affinity={ta} and threshold_similarity={ts}")
                    pbar.update(1)
                    continue

                # Compute performance metrics for the generated pairs
                number_of_pairs = len(groups)
                groups_perf_micro = get_performance(groups, r2_rmse, 'micro')
                groups_perf_macro = get_performance(groups, r2_rmse, 'macro')

                # Store the results for the current threshold combination
                tmp = [ta, ts, number_of_pairs]
                tmp.extend(groups_perf_micro)
                tmp.extend(groups_perf_macro)
                results.append(tmp)

                # Update the progress bar
                pbar.update(1)

    # Convert the results list into a DataFrame
    results = pd.DataFrame(results, columns=['threshold_affinity', 'threshold_similarity', 'number_of_pairs',
                                             'r2_micro', 'rmse_micro', 'r2_macro', 'rmse_macro'])
    return results


def get_results(drug_target_data, preds, file_name,
                save_preds = False,
                threshold_affinity=None, threshold_similarity=None):
    """
    Processes drug-target interaction data and predictions to compute metrics
    based on different affinity and similarity thresholds.

    Args:
        drug_target_data (str): Path to the CSV file containing drug-target interaction data.
        preds (str): Path to the file containing predicted values for drug-target pairs.
        file_name (str): The base name for saving output CSV files.
        threshold_affinity (list[float], optional): A list of affinity thresholds to evaluate.
                                                    Defaults to [0, 1, 1.5, 2, 2.5, 3, 3.5, 4].
        threshold_similarity (list[float], optional): A list of similarity thresholds to evaluate.
                                                      Defaults to [0, 0.1, 0.3, 0.5, 0.7, 0.9].

    Returns:
        DataFrame: A pandas DataFrame containing computed metrics across all threshold combinations.
    """

    # If no thresholds are provided, use default lists
    if threshold_affinity is None:
        threshold_affinity = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    if threshold_similarity is None:
        threshold_similarity = [0, 0.1, 0.3, 0.5, 0.7, 0.9]

    # Read the drug-target interaction data from the CSV file
    drug_target_data = pd.read_csv(drug_target_data)

    # Filter the data to include only the test set (where 'split' column equals 2)
    test_data = drug_target_data[drug_target_data['split'] == 2]

    # Remove the 'split' column from the test data, as it's no longer needed
    del test_data['split']

    test_data_cleaned = test_data[~test_data['SMILES'].apply(Chem.MolFromSmiles).isna()]
    # Load the predictions from the specified file
    preds = torch.load(preds)

    # Insert the predicted values as a new column in the test data
    test_data_cleaned.insert(4, 'predicted', preds)

    # Save the test data with predictions to a new CSV file
    if save_preds:
        test_data_cleaned.to_csv(f'../analysis/{file_name}.csv', index=False)

    # Compute performance metrics for all combinations of affinity and similarity thresholds
    results = get_total_metrics(test_data_cleaned,
                                threshold_affinity=threshold_affinity,
                                threshold_similarity=threshold_similarity)

    # Save the computed metrics to a CSV file
    results.to_csv(f'../analysis/preds/{file_name}_metrics.csv', index=False)

    # Return the DataFrame containing the computed metrics
    return results



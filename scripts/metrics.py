from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
from cliffs import get_similarity_matrix


def get_pairs(data, threshold_affinity=1, threshold_similarity=0.9):
    groups = []

    # Loop through each group in the DataFrame 'unpivoted' grouped by 'target'
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


def r2_rmse(g):
    r2 = r2_score(g['affinity'], g['predicted'])
    rmse = np.sqrt(mean_squared_error(g['affinity'], g['predicted']))
    return pd.Series({'r2': r2, 'rmse': rmse})


def get_performance(data, metric, averaging):
    df1 = data[['target', 'drug1', 'affinity1', 'predicted1']]
    df2 = data[['drug2', 'affinity2', 'predicted2']]

    new_df = pd.concat([df1.rename(columns={'drug1': 'drug', 'affinity1': 'affinity',
                                            'predicted1': 'predicted'}),
                        df2.rename(columns={'drug2': 'drug', 'affinity2': 'affinity',
                                            'predicted2': 'predicted'})], ignore_index=True)

    if averaging == 'micro':
        return metric(new_df)

    if averaging == 'macro':
        return new_df.groupby('target').apply(metric).mean()


def get_total_performance(data, metric, averaging):
    if averaging == 'micro':
        return metric(data)

    if averaging == 'macro':
        return data.groupby('target_id').apply(metric).mean()


def get_total_metrics(data, threshold_affinity: list[float], threshold_similarity: list[float]):
    results = []
    for ta in threshold_affinity:
        for ts in threshold_similarity:
            groups = get_pairs(data, ta, ts)

            if groups.empty:
                print(f"No data available for threshold_affinity={ta} and threshold_similarity={ts}")
                continue
            number_of_pairs = len(groups)
            groups_perf_micro = get_performance(groups, r2_rmse, 'micro')
            groups_perf_macro = get_performance(groups, r2_rmse, 'macro')
            tmp = [ta, ts, number_of_pairs]
            tmp.extend(groups_perf_micro)
            tmp.extend(groups_perf_macro)
            results.append(tmp)
    results = pd.DataFrame(results, columns=['threshold_affinity', 'threshold_similarity', 'number_of_pairs',
                                             'r2_micro', 'rmse_micro', 'r2_macro', 'rmse_macro'])
    return results

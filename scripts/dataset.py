import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from cliffs import get_similarity_matrix


def get_cliffs(data, threshold_affinity=1, threshold_similarity=0.9):
    pairs = []

    # Loop through each group in the data grouped by 'target'
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

        # Calculate the absolute difference in affinity between each pair of drugs
        affinity_diff = np.abs(d1.affinity.values - d2.affinity.values)

        # Determine cliff status: 1 if the difference is greater than the threshold, otherwise 0
        cliffs = (affinity_diff > threshold_affinity).astype(int)

        # Prepare a DataFrame for the current group with drug1, drug2, and cliff status
        current_pairs = pd.DataFrame({
            'drug1': d1.drug.values,
            'drug2': d2.drug.values,
            'cliff': cliffs,
            'target': g_name
        })

        pairs.append(current_pairs)  # Append the current group's pairs to the list

        # Concatenate all pairs into a single DataFrame
    pairs_df = pd.concat(pairs, ignore_index=True)

    return pairs_df


def split_data(data, split):
    if split == 'random':
        train, validation, test = random_split(data)
    elif split == 'compound-based':
        train, validation, test = compound_based_split(data)
    else:
        raise ValueError('Split should be either random or compound-based')

    train['split'] = 0
    validation['split'] = 1
    test['split'] = 2
    data = pd.concat([train, validation, test])
    return data


# Split the data randomly
def random_split(data):
    train, temp = train_test_split(data, test_size=0.3, random_state=42)
    validation, test = train_test_split(temp, test_size=(2/3), random_state=42)
    return train, validation, test


# Split the data compound-based
def compound_based_split(data):
    compounds = pd.concat([data['drug1'], data['drug2']]).unique()
    temp_compounds, test_compounds = train_test_split(compounds, test_size=0.1, random_state=42)
    test = data[data['drug1'].isin(test_compounds) | data['drug2'].isin(test_compounds)].copy()
    data_without_test = data[~data['drug1'].isin(test_compounds) &
                             ~data['drug2'].isin(test_compounds)].copy()

    compounds_2 = pd.concat([data_without_test['drug1'], data_without_test['drug2']]).unique()
    train_compounds, validation_compounds = train_test_split(compounds_2, test_size=0.1, random_state=2)
    validation = data_without_test[data_without_test['drug1'].isin(validation_compounds) |
                                   data_without_test['drug2'].isin(validation_compounds)].copy()

    train = data_without_test[~data_without_test['drug1'].isin(validation_compounds) &
                              ~data_without_test['drug2'].isin(validation_compounds)].copy()
    return train, validation, test


def majority_vote(data):
    majority_votes = data.groupby(['drug1', 'drug2'])['cliff'].apply(lambda x: np.bincount(x).argmax())
    vote_df = pd.DataFrame(majority_votes).reset_index()
    return vote_df


def at_least_once_AC(data):
    at_least_once = data.groupby(['drug1', 'drug2'])['cliff'].apply(lambda x: 1 if (x == 1).any() else 0)
    ac_df = pd.DataFrame(at_least_once).reset_index()
    return ac_df




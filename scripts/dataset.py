import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from cliffs import get_similarity_matrix
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed


def get_cliffs(data, threshold_affinity=1, threshold_similarity=0.9, task='classification'):
    pairs = []

    # Loop through each group in the data grouped by 'target'
    for g_name, group in tqdm(data.groupby('target', sort=False), desc="Processing targets"):
        # Calculate the similarity matrix for the drug molecules (related to specific target)
        sim = get_similarity_matrix(group.SMILES.to_list(), similarity=threshold_similarity)
        # Find non-zero elements in the similarity matrix, indicating pairs of similar drugs
        i, j = sim.nonzero()

        # Select corresponding rows from the affinity DataFrame for these drug pairs d1-d2
        d1 = group.iloc[i]
        d2 = group.iloc[j]

        # Calculate the absolute difference in affinity between each pair of drugs
        affinity_diff = np.abs(d1.affinity.values - d2.affinity.values)

        # Prepare a DataFrame for the current group with either cliff status or affinity difference
        if task == 'classification':
            cliffs = (affinity_diff > threshold_affinity).astype(int)
            current_pairs = pd.DataFrame({
                'drug1': d1['drug'].values,
                'drug2': d2['drug'].values,
                'smiles1': d1['SMILES'].values,
                'smiles2': d2['SMILES'].values,
                'cliff': cliffs,
                'target': g_name
            })
        elif task == 'regression':
            current_pairs = pd.DataFrame({
                'drug1': d1['drug'].values,
                'drug2': d2['drug'].values,
                'smiles1': d1['SMILES'].values,
                'smiles2': d2['SMILES'].values,
                'affinity_difference': affinity_diff,
                'target': g_name
            })

        pairs.append(current_pairs)  # Append the current group's pairs to the list

    # Concatenate all pairs into a single DataFrame
    pairs_df = pd.concat(pairs, ignore_index=True)

    return pairs_df


# Split the data randomly
def random_split_data(data):
    train, temp = train_test_split(data, test_size=0.3, random_state=42)
    validation, test = train_test_split(temp, test_size=(2/3), random_state=42)

    train['split'] = 0
    validation['split'] = 1
    test['split'] = 2
    data = pd.concat([train, validation, test])

    return data


# Split the data compound-based
def compound_based_split(data):
    # Get the unique drugs
    unique_drugs = data['drug'].unique()

    # Split the drugs into train, validation, and test sets
    train_drugs, temp_drugs = train_test_split(unique_drugs, test_size=0.3, random_state=42)
    val_drugs, test_drugs = train_test_split(temp_drugs, test_size=(2/3), random_state=42)

    # Create train, validation, and test sets
    train = data[data['drug'].isin(train_drugs)]
    validation = data[data['drug'].isin(val_drugs)]
    test = data[data['drug'].isin(test_drugs)]

    train['split'] = 0
    validation['split'] = 1
    test['split'] = 2
    data = pd.concat([train, validation, test])

    return data


# taken from DeepPurpose paper
def convert_y_unit(y, from_, to_):
    array_flag = False
    if isinstance(y, (int, float)):
        y = np.array([y])
        array_flag = True
    y = y.astype(float)
    # basis as nM
    if from_ == 'nM':
        y = y
    elif from_ == 'p':
        y = 10 ** (-y) / 1e-9

    if to_ == 'p':
        zero_idxs = np.where(y == 0.)[0]
        y[zero_idxs] = 1e-10
        y = -np.log10(y * 1e-9)
    elif to_ == 'nM':
        y = y

    if array_flag:
        return y[0]
    return y


def process_BindingDB(path='../data/BINDINGDB', df=None, y='Kd', binary=False,
                      convert_to_log=True, threshold=30, return_ids=False,
                      ids_condition='OR', harmonize_affinities=None):
    """
    :path: path to original BindingDB CSV/TSV data file. If None, then 'df' is expected.
    :param df: pre-loaded DataFrame
    :param y: type of binding affinity label. can be either 'Kd', 'IC50', 'EC50', 'Ki',
                or a list of strings with multiple choices.
    :param binary: whether to use binary labels
    :param convert_to_log: whether to convert nM units to P (log)
    :param threshold: threshold affinity for binary labels. can be a number or list
                of two numbers (low and high threshold)
    :param return_ids: whether to return drug and target ids
    :param ids_condition: keep samples for which drug AND/OR target IDs exist
    :param harmonize_affinities:  unify duplicate samples
                            'max' for choosing the instance with maximum affinity
                            'mean' for using the mean affinity of all duplicate instances
                            None to do nothing
    """
    if path is not None and not os.path.exists(path):
        os.makedirs(path)

    if df is not None:
        print('Loading Dataset from the pandas input...')
    elif path is not None:
        print('Loading Dataset from path...')
        df = pd.read_csv(path, sep='\t', on_bad_lines='skip')
    else:
        ValueError("Either 'df' of 'path' must be provided")

    print('Beginning Processing...')
    df = df[df['Number of Protein Chains in Target (>1 implies a multichain complex)'] == 1.0]
    df = df[df['Ligand SMILES'].notnull()]

    idx_str = []
    yy = y
    if isinstance(y, str):
        yy = [y]
    for y in yy:
        if y == 'Kd':
            idx_str.append('Kd (nM)')
        elif y == 'IC50':
            idx_str.append('IC50 (nM)')
        elif y == 'Ki':
            idx_str.append('Ki (nM)')
        elif y == 'EC50':
            idx_str.append('EC50 (nM)')
        else:
            print('select Kd, Ki, IC50 or EC50')

    if len(idx_str) == 1:
        df_want = df[df[idx_str[0]].notnull()]
    else:  # select multiple affinity measurements.
        # keep rows for which at least one of the columns in the idx_str list is not null
        df_want = df.dropna(thresh=1, subset=idx_str)

    df_want = df_want[['BindingDB Ligand Name', 'BindingDB Reactant_set_id', 'Ligand InChI', 'Ligand SMILES',
                       'PubChem CID', 'UniProt (SwissProt) Primary ID of Target Chain',
                       'Target Name'] + idx_str]

    for y in idx_str:
        df_want[y] = df_want[y].str.replace('>', '')
        df_want[y] = df_want[y].str.replace('<', '')
        df_want[y] = df_want[y].astype(float)

    # Harmonize into single label using the mean of existing labels:
    df_want['Label'] = df_want[idx_str].mean(axis=1, skipna=True)

    df_want.rename(columns={'BindingDB Reactant_set_id': 'ID',
                            'BindingDB Ligand Name': 'Drug',
                            'Ligand SMILES': 'SMILES',
                            'Ligand InChI': 'InChI',
                            'PubChem CID': 'PubChem_ID',
                            'UniProt (SwissProt) Primary ID of Target Chain': 'UniProt_ID',
                            'Target Name': 'Target Sequence'},
                   inplace=True)

    # have at least uniprot or pubchem ID
    if ids_condition == 'OR':
        df_want = df_want[df_want.PubChem_ID.notnull() | df_want.UniProt_ID.notnull()]
    elif ids_condition == 'AND':
        df_want = df_want[df_want.PubChem_ID.notnull() & df_want.UniProt_ID.notnull()]
    else:
        ValueError("ids_condition must be set to 'OR' or 'AND'")

    df_want = df_want[df_want.InChI.notnull()]

    df_want = df_want[df_want.Label <= 10000000.0]
    print('There are ' + str(len(df_want)) + ' drug target pairs.')

    if harmonize_affinities is not None:
        df_want = df_want[['PubChem_ID', 'SMILES', 'UniProt_ID', 'Target Sequence', 'Label']]
        if harmonize_affinities.lower() == 'max_affinity':
            df_want = df_want.groupby(['PubChem_ID', 'SMILES', 'UniProt_ID', 'Target Sequence']).Label.agg(
                min).reset_index()
        if harmonize_affinities.lower() == 'mean':
            df_want = df_want.groupby(['PubChem_ID', 'SMILES', 'UniProt_ID', 'Target Sequence']).Label.agg(
                np.mean).reset_index()

    if binary:
        print(
            'Default binary threshold for the binding affinity scores are 30, you can adjust it by using the "threshold" parameter')
        if isinstance(threshold, Sequence):
            # filter samples with affinity values between the thresholds
            df_want = df_want[(df_want.Label < threshold[0]) | (df_want.Label > threshold[1])]
        else:  # single threshold
            threshold = [threshold]
        y = [1 if i else 0 for i in df_want.Label.values < threshold[0]]
    else:
        if convert_to_log:
            print('Default set to logspace (nM -> p) for easier regression')
            y = convert_y_unit(df_want.Label.values, 'nM', 'p')
        else:
            y = df_want.Label.values

    if return_ids:
        return df_want.SMILES.values, df_want['Target Sequence'].values, np.array(y), df_want['PubChem_ID'].values, \
        df_want['UniProt_ID'].values
    return df_want['Drug'].values, df_want.SMILES.values, df_want['Target Sequence'].values, np.array(y)




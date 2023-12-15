import numpy as np
from functools import cache
from Levenshtein import distance as levenshtein
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework, GetScaffoldForMol
from rdkit.DataStructs import TanimotoSimilarity as ts
from tqdm import trange


def get_similarity_matrix(smiles, radius: int = 2, nBits: int = 1024, similarity: float = .9,
                          levenshtein_sim=True, structure_sim=True, scaffold_sim=True):
    molecule = cache(MolFromSmiles)

    @cache
    def fingerprint(smi):
        return GetMorganFingerprintAsBitVect(molecule(smi), radius=radius, nBits=nBits)

    @cache
    def scaffold_fingerprint(smi):
        mol = molecule(smi)
        try:
            skeleton = GraphFramework(mol)
        except Exception:
            skeleton = GetScaffoldForMol(mol)
        return GetMorganFingerprintAsBitVect(skeleton, radius=radius, nBits=nBits)

    size = len(smiles)
    matrix = np.zeros([size, size], dtype=bool)
    # Calculate upper triangle of matrix
    for i in trange(size - 1):
        for j in range(i + 1, size):
            # similarity = 1 - distance
            if (levenshtein_sim and
                    1 - levenshtein(smiles[i], smiles[j]) / max(len(smiles[i]), len(smiles[j])) > similarity):
                matrix[i, j] = True
            elif structure_sim and ts(fingerprint(smiles[i]), fingerprint(smiles[j])) > similarity:
                matrix[i, j] = True
            elif (scaffold_sim and
                    ts(scaffold_fingerprint(smiles[i]), scaffold_fingerprint(smiles[j])) > similarity):
                matrix[i, j] = True
    return matrix



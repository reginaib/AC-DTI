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
    # Cache the MolFromSmiles function for efficient molecule conversion
    molecule = cache(MolFromSmiles)

    @cache
    def fingerprint(smi):
        # Generate a fingerprint for a molecule for structural similarity comparison
        return GetMorganFingerprintAsBitVect(molecule(smi), radius=radius, nBits=nBits)

    @cache
    def scaffold_fingerprint(smi):
        # Generate a scaffold fingerprint for a molecule
        mol = molecule(smi)
        try:
            # Try to create a generic framework (skeleton) of the molecule
            skeleton = GraphFramework(mol)
        except Exception:
            # If the generic framework fails, use the standard scaffold
            skeleton = GetScaffoldForMol(mol)
        return GetMorganFingerprintAsBitVect(skeleton, radius=radius, nBits=nBits)

    size = len(smiles)

    # Create a similarity matrix. by default all molecules are not similar
    matrix = np.zeros([size, size], dtype=bool)
    # Calculate the upper triangle of the similarity matrix
    for i in trange(size - 1):
        for j in range(i + 1, size):
            # Check for similarity based on different criteria
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



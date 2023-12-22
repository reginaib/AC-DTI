import numpy as np
from functools import cache
from Levenshtein import distance
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework, GetScaffoldForMol
from rdkit.DataStructs import TanimotoSimilarity
from tqdm import trange


# Cache the MolFromSmiles function for efficient molecule conversion
molecule = cache(MolFromSmiles)


# Cache the result of the functions to avoid redundant  calculations for the same pair of SMILES
@cache
def levenshtein(smiles1, smiles2):
    # Calculate the Levenshtein distance between two SMILES strings
    return 1 - distance(smiles1, smiles2) / max(len(smiles1), len(smiles2))


@cache
def tanimoto(smiles1, smiles2, radius, nBits):
    # Calculate Tanimoto similarity between two molecules based on their fingerprints
    return TanimotoSimilarity(fingerprint(smiles1, radius, nBits), fingerprint(smiles2, radius, nBits))


@cache
def scaffold_tanimoto(smiles1, smiles2, radius, nBits):
    # Calculate Tanimoto structure similarity between molecular scaffolds of two molecules
    return TanimotoSimilarity(scaffold_fingerprint(smiles1, radius, nBits),
                              scaffold_fingerprint(smiles2, radius, nBits))


# Cache the result of the functions to avoid redundant fingerprints generation for the same SMILES string
@cache
def fingerprint(smi, radius, nBits):
    # Generate a fingerprint for a molecule for structural similarity comparison
    return GetMorganFingerprintAsBitVect(molecule(smi), radius=radius, nBits=nBits)


# Cache the result of the functions to avoid redundant scaffold fingerprints generation for the same SMILES string
@cache
def scaffold_fingerprint(smi, radius, nBits):
    # Generate a scaffold fingerprint for a molecule
    mol = molecule(smi)
    try:
        # Try to create a generic framework (skeleton) of the molecule
        skeleton = GraphFramework(mol)
    except Exception:
        # If the generic framework fails, use the standard scaffold
        skeleton = GetScaffoldForMol(mol)
    return GetMorganFingerprintAsBitVect(skeleton, radius=radius, nBits=nBits)


def get_similarity_matrix(smiles, radius: int = 2, nBits: int = 1024, similarity: float = .9,
                          levenshtein_sim=True, structure_sim=True, scaffold_sim=True):
    size = len(smiles)

    # Create a similarity matrix. by default all molecules are not similar
    matrix = np.zeros([size, size], dtype=bool)
    # Calculate the upper triangle of the similarity matrix
    for i in trange(size - 1):
        for j in range(i + 1, size):
            # Check for similarity based on different criteria
            if levenshtein_sim and levenshtein(smiles[i], smiles[j]) > similarity:
                matrix[i, j] = True
            elif structure_sim and tanimoto(smiles[i], smiles[j], radius, nBits) > similarity:
                matrix[i, j] = True
            elif scaffold_sim and scaffold_tanimoto(smiles[i], smiles[j], radius, nBits) > similarity:
                matrix[i, j] = True
    return matrix

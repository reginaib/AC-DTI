import numpy as np
from functools import cache
from Levenshtein import distance
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework, GetScaffoldForMol
from rdkit.DataStructs import TanimotoSimilarity


RDLogger.DisableLog('rdApp.error')


# Cache the MolFromSmiles function for efficient molecule conversion

@cache
def molecule(smi):
    try:
        return MolFromSmiles(smi)
    except:
        return


@cache
def levenshtein(smiles1, smiles2):
    # Calculate the Levenshtein distance between two SMILES strings
    return 1 - distance(smiles1, smiles2) / max(len(smiles1), len(smiles2))


def tanimoto(smiles1, smiles2, radius, nBits):
    # Calculate Tanimoto similarity between two molecules based on their fingerprints
    fp1 = fingerprint(smiles1, radius, nBits)
    fp2 = fingerprint(smiles2, radius, nBits)

    # Check if either fingerprint is None
    if fp1 is None or fp2 is None:
        return 0

    return TanimotoSimilarity(fp1, fp2)


@cache
def scaffold_tanimoto(smiles1, smiles2, radius, nBits):
    # Calculate Tanimoto structure similarity between molecular scaffolds of two molecules
    scaffold_fp1 = scaffold_fingerprint(smiles1, radius, nBits)
    scaffold_fp2 = scaffold_fingerprint(smiles2, radius, nBits)

    # Check if either scaffold fingerprint is None
    if scaffold_fp1 is None or scaffold_fp2 is None:
        return 0

    return TanimotoSimilarity(scaffold_fp1, scaffold_fp2)


@cache
def fingerprint(smi, radius, nBits):
    try:
        mol = molecule(smi)
        # Check if molecule is None (invalid SMILES)
        if mol is None:
            return None
        return GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    except Exception as e:
        # Handle any exception that occurs during fingerprint generation
        print(f"Error processing SMILES '{smi}': {e}")
        return None


@cache
def scaffold_fingerprint(smi, radius, nBits):
    try:
        # Generate a scaffold fingerprint for a molecule
        mol = molecule(smi)
        # Check if molecule is None (invalid SMILES)
        if mol is None:
            return None

        try:
            # Try to create a generic framework (skeleton) of the molecule
            skeleton = GraphFramework(mol)
        except Exception:
            # If the generic framework fails, use the standard scaffold
            skeleton = GetScaffoldForMol(mol)

        return GetMorganFingerprintAsBitVect(skeleton, radius=radius, nBits=nBits)
    except (ValueError, RuntimeError):
        # Skip molecules with explicit valence errors
        return None


def get_similarity_matrix(smiles, radius: int = 2, nBits: int = 1024, similarity: float = .9,
                          levenshtein_sim=True, structure_sim=True, scaffold_sim=True):
    size = len(smiles)

    # Create a similarity matrix. by default all molecules are not similar
    matrix = np.zeros([size, size], dtype=bool)
    # Calculate the upper triangle of the similarity matrix
    for i in range(size - 1):
        for j in range(i + 1, size):
            # Check for similarity based on different criteria
            if levenshtein_sim and levenshtein(smiles[i], smiles[j]) > similarity:
                matrix[i, j] = True
            elif structure_sim and tanimoto(smiles[i], smiles[j], radius, nBits) > similarity:
                matrix[i, j] = True
            elif scaffold_sim and scaffold_tanimoto(smiles[i], smiles[j], radius, nBits) > similarity:
                matrix[i, j] = True
    return matrix

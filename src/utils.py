from __future__ import annotations

import random

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

#calulate morgan 2 fingerprints for all compounds
def calculate_fingerprints(smiles_list, num_bits=2048, radius=2):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=num_bits)
    fp_array = []
    for tmp_smi in smiles_list:
        mol = Chem.MolFromSmiles(tmp_smi)
        fp = mfpgen.GetFingerprintAsNumPy(mol)
        fp_array.append(fp)
    fp_array = np.array(fp_array)
    
    return fp_array

def get_rdkit_descriptors(smiles_list):

    desc_list = []
    for tmp_smi in smiles_list:
        res = []
        mol = Chem.MolFromSmiles(tmp_smi)
        for nm,fn in Descriptors._descList:
            # some of the descriptor fucntions can throw errors if they fail, catch those here:
            try:
                val = fn(mol)
            except:
                # print the error message:
                val = np.nan
            res.append(val)
        desc_list.append(res)
    
    desc_array = np.array(desc_list)
    return desc_array


# Balanced scaffold splitting
def balanced_scaffold_split(smiles, frac_train=0.8, frac_val=0.1, seed=42):
    random.seed(seed)
    scaffolds = {}
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else f"invalid_{i}"
        scaffolds.setdefault(scaf, []).append(i)
    groups = list(scaffolds.values())
    print(f"Number of unique scaffolds: {len(groups)}")
    random.shuffle(groups)
    n = len(smiles)
    train_target, val_target, test_target = int(frac_train*n), int(frac_val*n), n - int(frac_train*n) - int(frac_val*n)
    train, val, test = [], [], []
    for g in groups:
        if len(test + g) < test_target:
            test += g
        elif len(val + g) < val_target:
            val += g
        else:
            train += g
    return train, val, test

# Cluster-based butina splitting
def cluster_based_split(smiles, frac_train=0.8, frac_val=0.1, random_seed=42, distance_threshold=0.3):

    random.seed(random_seed)
    fps = [rdFingerprintGenerator.GetRDKitFPGenerator().GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    clusters = Butina.ClusterData(dists, len(fps), distance_threshold, isDistData=True)
    cluster_list = list(clusters)
    print(f"Number of clusters: {len(cluster_list)}")
    random.shuffle(cluster_list)
    n = len(smiles)
    train_target, val_target = int(frac_train*n), int(frac_val*n)
    train, val, test = [], [], []
    for clust in cluster_list:
        if len(val) < val_target and (len(val)+len(clust)) <= val_target:
            val += clust
        elif len(train) < train_target and (len(train)+len(clust)) <= train_target:
            train += clust
        else:
            test += clust
    return train, val, test
from __future__ import annotations

import random

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from numba import njit, prange
    

#------------------------------------------------
def get_rdkit_descriptors(smiles_list):

    desc_list = []
    for tmp_smi in smiles_list:
        res = []
        mol = Chem.MolFromSmiles(tmp_smi)
        for _, fn in Descriptors._descList:
            # some of the descriptor fucntions can throw errors if they fail, catch those here:
            try:
                val = fn(mol)
            except Exception:
                # print the error message:
                val = np.nan
            res.append(val)
        desc_list.append(res)
    
    desc_array = np.array(desc_list)
    return desc_array

#***********************************
#***** FINGERPRINT FUNCTIONS *******
#***********************************

#------------------------------------------------
def calculate_fingerprints(smiles_list, num_bits=2048, radius=2):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=num_bits)
    fp_array = []
    for tmp_smi in smiles_list:
        mol = Chem.MolFromSmiles(tmp_smi)
        fp = mfpgen.GetFingerprintAsNumPy(mol)
        fp_array.append(fp)
    fp_array = np.array(fp_array)
    return fp_array


#---------------------------------------------------------------
#numba implementation of Tanimoto distance
@njit(nopython=True)
def tanimoto_distance(v1, v2):
    #Calculates tanimoto distance for two bit vectors
    bit_sum = v1 + v2
    bitwise_and = count_loop_equals2(bit_sum)
    bitwise_or = count_loop_bigger0(bit_sum)
    
    jaccard_distance = 1 - (bitwise_and / bitwise_or)
    
    return jaccard_distance


#--------------------------------------------------------------
@njit()
def count_loop_equals2(a):
    s = 0
    for i in a:
        if i == 2:
            s += 1
    return s

#--------------------------------------------------------------
@njit()
def count_loop_bigger0(a):
    s = 0
    for i in a:
        if i > 0:
            s += 1
    return s


#----------------------------------------------------------------
@njit(nopython=True, parallel=True)
def get_dists_between_two_sets(fp_array_1, fp_array_2):
    
    num_fp_1 = fp_array_1.shape[0]
    num_fp_2 = fp_array_2.shape[0]
    
    dist_matrix = np.zeros(num_fp_1, num_fp_2, dtype=np.float32)
    
    #calculate distances
    for fp_ind_1 in prange(num_fp_1):
        
        tmp_fp_1 = fp_array_1[fp_ind_1, :]
        
        for fp_ind_2 in prange(num_fp_2):
            
            tmp_fp_2 = fp_array_2[fp_ind_2, :]
            
            #Calculates tanimoto distance for two bit vectors
            bit_sum = tmp_fp_1 + tmp_fp_2
              
            bitwise_and = count_loop_equals2(bit_sum)
            bitwise_or = count_loop_bigger0(bit_sum)
    
            tmp_dist = np.float32(1 - (bitwise_and / float(bitwise_or)))
        
            dist_matrix[fp_ind_1, fp_ind_2] = tmp_dist
                 
    return dist_matrix


#----------------------------------------------------------------
#get nearest neighbors from distance matrix
def get_nearest_neighbor_distances(fp_array_1, fp_array_2):
    
    dist_matrix = get_dists_between_two_sets(fp_array_1, fp_array_2)
    
    nearest_neighbor_dists = np.min(dist_matrix, axis=1)
    
    return nearest_neighbor_dists


#*****************************************************
#**************** SPLIT FUNCTIONS ********************
#*****************************************************

#---------------------------------------------------------------
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
    _, val_target, test_target = int(frac_train*n), int(frac_val*n), n - int(frac_train*n) - int(frac_val*n)
    train, val, test = [], [], []
    for g in groups:
        if len(test + g) < test_target:
            test += g
        elif len(val + g) < val_target:
            val += g
        else:
            train += g
    return train, val, test

#---------------------------------------------------------------
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
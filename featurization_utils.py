### for proteins
token_dict = {'_PAD':0, '_GO':1, '_EOS':2, '_UNK':3, 'A': 4, 'R': 5, 'N': 6, 'D': 7, 'C': 8, 'Q': 9, 'E': 10, 'G': 11, 'H': 12, 'I': 13, 'L': 14, 'K': 15, 'M': 16, 'F': 17, 'P': 18, 'S': 19, 'T': 20, 'W': 21, 'Y': 22, 'V': 23, 'X': 24, 'U': 25, 'O': 26, 'B': 27, 'Z': 28}


### for compounds
import pandas as pd
import numpy as np
from rdkit import Chem
import rdkit.Chem.rdPartialCharges as rdPartialCharges
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
import scipy.sparse as sps


mydict = {'C':0,'N':1,'O':2,'S':3,'F':4,'Si':5,'Cl':6,'P':7,'Br':8,'I':9,'B':10,'Unknown':11,'_PAD':12}
mydict_hal_don = {'C':0,'N':0,'O':0,'S':0,'F':1,'Si':0,'Cl':2,'P':0,'Br':3,'I':4,'B':0,'Unknown':0,'_PAD':0}
mydict_hybrid = {Chem.rdchem.HybridizationType.SP:0,Chem.rdchem.HybridizationType.SP2:1,Chem.rdchem.HybridizationType.SP3:2,Chem.rdchem.HybridizationType.SP3D:3,Chem.rdchem.HybridizationType.SP3D2:4}


def one_hot_enc(size,ind):
    x = np.zeros(size)
    x[ind] = 1
    return x

def get_ar_mol(mol,MAX_size):
    atom_list = []
    for a in mol.GetAtoms():
        m = a.GetIsAromatic()
        atom_list.append(m)

    if len(atom_list) < MAX_size:
       pad = [0] * (MAX_size - len(atom_list))
       atom_list = atom_list+pad

    return np.array(atom_list, np.int32)

def get_pol_mol(mol,MAX_size):
    atom_list = []
    for a in mol.GetAtoms():
        m = a.GetProp('_GasteigerCharge')
        atom_list.append(m)

    if len(atom_list) < MAX_size:
       pad = [0] * (MAX_size - len(atom_list))
       atom_list = atom_list+pad

    atom_list = np.array(atom_list, np.float32)
    # 2 some compound atoms do not have the properties, filter the nan and inf values
    if np.isnan(atom_list.sum()) or np.isinf(atom_list.sum()):
        atom_list[np.isnan(atom_list)] = 0
        atom_list[np.isinf(atom_list)] = 0
    return atom_list

def get_charge_mol(mol,MAX_size):
    atom_list = []
    for a in mol.GetAtoms():
        m = a.GetFormalCharge()
        atom_list.append(m)

    if len(atom_list) < MAX_size:
       pad = [0] * (MAX_size - len(atom_list))
       atom_list = atom_list+pad

    return np.array(atom_list, np.float32)

def get_sym_mol(mol,mydict,MAX_size):
    atom_list = []
    for a in mol.GetAtoms():
        m = a.GetSymbol()
        if m in mydict:
           atom_list.append(mydict[m])
        else:
           atom_list.append(mydict['Unknown'])

    if len(atom_list) < MAX_size:
       pad = [mydict['_PAD']] * (MAX_size - len(atom_list))
       atom_list = atom_list+pad

    return np.array(atom_list, np.int32)

def get_sym_mol_one_hot(mol,mydict,MAX_size):
    atom_list = []
    for a in mol.GetAtoms():
        m = a.GetSymbol()
        if m in mydict:
           atom_list.append(one_hot_enc(len(mydict),mydict[m]))
        else:
           atom_list.append(one_hot_enc(len(mydict),mydict['Unknown']))

    if len(atom_list) < MAX_size:
       pad = [one_hot_enc(len(mydict),-1)] * (MAX_size - len(atom_list))
       atom_list = atom_list+pad

    return np.array(atom_list, np.int32)

def get_degree_mol(mol,MAX_size):
    atom_list = []
    for a in mol.GetAtoms():
        m = a.GetDegree()
        if m <= 5:
            atom_list.append(one_hot_enc(6,m))
        else: # 1 some atoms have degree > 6, then encode no degree information
            atom_list.append(np.ones(6)/6)

    if len(atom_list) < MAX_size:
       pad = [one_hot_enc(6,0)] * (MAX_size - len(atom_list))
       atom_list = atom_list+pad

    return np.array(atom_list, np.float32)

def get_numH_mol(mol,MAX_size):
    atom_list = []
    for a in mol.GetAtoms():
        m = a.GetTotalNumHs()
        atom_list.append(one_hot_enc(5,m))

    if len(atom_list) < MAX_size:
       pad = [one_hot_enc(5,0)] * (MAX_size - len(atom_list))
       atom_list = atom_list+pad

    return np.array(atom_list, np.float32)

def get_implicitValence_mol(mol,MAX_size):
    atom_list = []
    for a in mol.GetAtoms():
        m = a.GetImplicitValence()
        atom_list.append(one_hot_enc(6,m))

    if len(atom_list) < MAX_size:
       pad = [one_hot_enc(6,0)] * (MAX_size - len(atom_list))
       atom_list = atom_list+pad

    return np.array(atom_list, np.float32)

def get_numRadicalElen_mol(mol,MAX_size):
    atom_list = []
    for a in mol.GetAtoms():
        m = a.GetNumRadicalElectrons()
        atom_list.append(m)

    if len(atom_list) < MAX_size:
       pad = [0] * (MAX_size - len(atom_list))
       atom_list = atom_list+pad

    return np.array(atom_list, np.float32)

def get_hybridization_mol(mol,mydict,MAX_size):
    atom_list = []
    for a in mol.GetAtoms():
        m = a.GetHybridization()
        if m in mydict.keys():
            atom_list.append(one_hot_enc(len(mydict)+1,mydict[m]))
        else: # 3 some hybridization type is missing
            atom_list.append(np.ones(len(mydict)+1)/(len(mydict)+1))

    if len(atom_list) < MAX_size:
       pad = [one_hot_enc(len(mydict)+1,-1)] * (MAX_size - len(atom_list))
       atom_list = atom_list+pad

    return np.array(atom_list, np.int32)

def get_hyd_mol(mol,factory,MAX_size):
    atom_list = []
    for a in mol.GetAtoms():
        m = a.GetSymbol()
        atom_list.append(m)

    feats = factory.GetFeaturesForMol(mol)
    final_feat = np.zeros((MAX_size,2))
    for i in range(len(feats)):
       t = feats[i].GetType()
       if t == "SingleAtomDonor":
          final_feat[feats[i].GetAtomIds()[0],0] = 1
       elif t == "SingleAtomAcceptor":
          final_feat[feats[i].GetAtomIds()[0],1] = 1

    if len(atom_list) < MAX_size:
       final_feat[len(atom_list):,:]= 0

    return final_feat


fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

def read_graph(smiles, MAX_size):
    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)
    rdPartialCharges.ComputeGasteigerCharges(mol)

    temp = np.reshape(get_ar_mol(mol,MAX_size),(MAX_size,1))
    temp2 = np.reshape(get_pol_mol(mol,MAX_size),(MAX_size,1))
    temp = np.concatenate((temp,temp2),axis=1)
    temp2 = np.reshape(get_charge_mol(mol,MAX_size),(MAX_size,1))
    temp = np.concatenate((temp,temp2),axis=1)
    temp2 = np.reshape(get_sym_mol_one_hot(mol,mydict,MAX_size),(MAX_size,len(mydict)))
    temp = np.concatenate((temp,temp2),axis=1)
    temp2 = np.reshape(get_hyd_mol(mol,factory,MAX_size),(MAX_size,2))
    temp = np.concatenate((temp,temp2),axis=1)
    temp2 = np.reshape(get_sym_mol(mol,mydict_hal_don,MAX_size),(MAX_size,1))
    temp = np.concatenate((temp,temp2),axis=1)
    temp2 = np.reshape(get_degree_mol(mol,MAX_size),(MAX_size,6))
    temp = np.concatenate((temp,temp2),axis=1)
    temp2 = np.reshape(get_numH_mol(mol,MAX_size),(MAX_size,5))
    temp = np.concatenate((temp,temp2),axis=1)
    temp2 = np.reshape(get_implicitValence_mol(mol,MAX_size),(MAX_size,6))
    temp = np.concatenate((temp,temp2),axis=1)
    temp2 = np.reshape(get_numRadicalElen_mol(mol,MAX_size),(MAX_size,1))
    temp = np.concatenate((temp,temp2),axis=1)
    temp2 = np.reshape(get_hybridization_mol(mol,mydict_hybrid,MAX_size),(MAX_size,len(mydict_hybrid)+1))
    temp = np.concatenate((temp,temp2),axis=1)

    adja_mat = Chem.GetAdjacencyMatrix(mol)
    adj_temp = []
    for adja in adja_mat:
        if len(adja) < MAX_size:
           pad = [0]*(MAX_size - len(adja))
           adja = np.array(list(adja)+pad,np.int32)
        adj_temp.append(adja)

    cur_len = len(adj_temp)
    for i in range(MAX_size - cur_len):
        adja =np.array( [0]*MAX_size,np.int32)
        adj_temp.append(adja)

    adj_temp = adj_temp + np.eye(MAX_size) # A_hat = A + I
    # return sps.csr_matrix(temp.reshape(1, -1)).astype('float32'), sps.csr_matrix(adj_temp.reshape(1, -1)).astype('int8'), source
    return temp.astype('float32'), adj_temp.astype('float32')

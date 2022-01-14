## importing part
from __future__ import division, print_function, absolute_import
# import statsmodels.api as sm
import gzip
import os
import re
import tarfile
import math
import random
import sys
import time
import logging
import numpy as np
import math
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
from Bio.PDB import *
"""

from tensorflow.python.platform import gfile
import tensorflow as tf
import keras
from keras.layers import BatchNormalization,Input,Reshape,Embedding,GRU,LSTM,Conv1D,Conv2D,LeakyReLU,MaxPooling1D,GlobalMaxPooling2D
from keras.layers import concatenate,Dropout,Dense,LeakyReLU,TimeDistributed,MaxPooling2D,add,Activation,SeparableConv2D,multiply
from keras import regularizers
from keras.optimizers import SGD,Adam
from keras.losses import mean_squared_error
from keras.models import Model, load_model
import keras.backend as K
from keras.activations import relu
from keras.engine.topology import Layer
from keras.callbacks import ModelCheckpoint,TensorBoard

from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem, ChemicalFeatures, rdchem
from rdkit import RDConfig
import rdkit.Chem.rdPartialCharges as rdPartialCharges

from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve,average_precision_score
from functools import cmp_to_key


################################################ Hyper-parameters #################################

#group_size = int(sys.argv[5])
#num_group = int(sys.argv[6])
#dataset = str(sys.argv[7])
dataset = "split_data"
#### data and vocabulary
interaction_dir = "../data/merged_data/interaction_shifted/"
pdb_dir = "../data/merged_data/pdb_used/"

map_dir = "prediction_map"

data_dir="../data/merged_data/" + dataset
vocab_size_protein=24
vocab_size_compound=15
vocab_protein="vocab_protein_24"
cid_dir = 'Kd/'
batch_size = 64
image_dir = "Kd/"

GRU_size_prot=256
GRU_size_drug=256
dev_perc=0.1
os.system('mkdir ' + map_dir)

## dictionary compound
index_aa = {0:'_PAD',1:'_GO',2:'_EOS',3:'_UNK',4:'A',5:'R',6:'N',7:'D',8:'C',9:'Q',10:'E',11:'G',12:'H',13:'I',14:'L',15:'K',16:'M',17:'F',18:'P',19:'S',20:'T',21:'W',22:'Y',23:'V'}
pdb2single = {"GLY":"G","ALA":"A","SER":"S","THR":"T","CYS":"C","VAL":"V","LEU":"L","ILE":"I","MET":"M","PRO":"P",\
"PHE":"F","TYR":"Y","TRP":"W","ASP":"D","GLU":"E","ASN":"N","GLN":"Q","HIS":"H","LYS":"K","ARG":"R", "UNK":"X",\
 "SEC":"U","PYL":"O","MSE":"M","CAS":"C","SGB":"S","CGA":"E","TRQ":"W","TPO":"T","SEP":"S","CME":"C","FT6":"W","OCS":"C","SUN":"S","SXE":"S"}
dict_atom = {'C':0,'N':1,'O':2,'S':3,'F':4,'Si':5,'Cl':6,'P':7,'Br':8,'I':9,'B':10,'Unknown':11,'_PAD':12}
dict_atom_hal_don = {'C':0,'N':0,'O':0,'S':0,'F':1,'Si':0,'Cl':2,'P':0,'Br':3,'I':4,'B':0,'Unknown':0,'_PAD':0}
dict_atom_hybridization = {Chem.rdchem.HybridizationType.SP:0,Chem.rdchem.HybridizationType.SP2:1,Chem.rdchem.HybridizationType.SP3:2,Chem.rdchem.HybridizationType.SP3D:3,Chem.rdchem.HybridizationType.SP3D2:4}


## Padding part
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
_WORD_SPLIT = re.compile(b"(\S)")
_WORD_SPLIT_2 = re.compile(b",")
_DIGIT_RE = re.compile(br"\d")
group_size = 40
num_group = 25
feature_num = len(dict_atom) + 24+ len(dict_atom_hybridization) + 1
full_prot_size = group_size * num_group
prot_max_size = group_size * num_group
comp_max_size = 56




## functions
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

    return np.array(atom_list, np.float32)

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
        atom_list.append(one_hot_enc(6,m))

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
        atom_list.append(one_hot_enc(len(mydict)+1,mydict[m]))

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

def read_graph(source_path,MAX_size,mydict,mydict_hal_don,mydict_hybrid):
  Vertex = []
  Adj = []
  fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
  factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

  with tf.gfile.GFile(source_path, mode="r") as source_file:
      source = source_file.readline().strip()
      counter = 0
      while source:
        mol = Chem.MolFromSmiles(source)
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
        Vertex.append(temp)
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
        deg = np.power(np.sum(adj_temp,axis=1),-0.5)
        deg_diag = np.diag(deg)
        adj = np.matmul(deg_diag,adj_temp)
        adj = np.matmul(adj,deg_diag) # normalized
        Adj.append(adj_temp)
        source = source_file.readline().strip()

  Vertex = np.asarray(Vertex)
  #Vertex_ar = np.asarray(Vertex_ar)
  #Vertex_pol = np.asarray(Vertex_pol)
  #Vertex_charge = np.asarray(Vertex_charge)
  #Vertex_hyd = np.asarray(Vertex_hyd)
  #Vertex_hal_acc = np.asarray(Vertex_hal_acc)
  #Vertex_hal_don = np.asarray(Vertex_hal_don)
  Adj = np.asarray(Adj)
  #return Vertex,Vertex_ar,Vertex_pol,Vertex_charge,Vertex_hyd,Vertex_hal_acc,Vertex_hal_don,Adj,Vertex_length
  return Vertex,Adj


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    #if condition ==0:
    #    l = _WORD_SPLIT.split(space_separated_fragment)
    #    del l[0::2]
    #elif condition == 1:
    #    l = _WORD_SPLIT_2.split(space_separated_fragment)
    l = _WORD_SPLIT.split(space_separated_fragment)
    del l[0::2]
    words.extend(l)
  return [w for w in words if w]

def sentence_to_token_ids(sentence, vocabulary,normalize_digits=False):

  words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)



def data_to_token_ids(data_path, target_path, vocabulary_path,normalize_digits=False):
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")



def read_data(source_path,MAX_size,group_size):
  data_set = []
  mycount=0
  with tf.gfile.GFile(source_path, mode="r") as source_file:
      source = source_file.readline()
      counter = 0
      while source:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()

        source_ids = [int(x) for x in source.split()]
        if len(source_ids) < MAX_size:
           pad = [PAD_ID] * (MAX_size - len(source_ids))
           #data_set.append(list(source_ids + pad)) #MK
      
           source_ids = list(source_ids + pad)
        elif len(source_ids) == MAX_size:
           #data_set.append(list(source_ids))

           #MK add alaki
           source_ids = list(source_ids)
        else:
           print("there is a data with length bigger than the max\n")
           print(len(source_ids))
        count=0
        temp=[]
        temp2=[]
        for x in source_ids:
          count=count+1
          if count < group_size+1:
            temp.append(x)
          if count == group_size+1:
            count=1
            temp2.append(temp)
            temp=[]
            temp.append(x)

        temp2.append(temp)
        data_set.append(temp2) 

        mycount=mycount+1
        source = source_file.readline()
  return data_set



def prepare_data(data_dir, train_path, vocabulary_size,vocab,max_size,group_size):
  vocab_path = os.path.join(data_dir, vocab)
  train_ids_path = train_path + (".ids%d" % vocabulary_size)
  data_to_token_ids(train_path, train_ids_path, vocab_path)
  train_set = read_data(train_ids_path,max_size,group_size)
  
  return train_set

def read_labels(path):
    x = []
    f = open(path, "r") 
    for line in f:
         if (line[0]=="<")or(line[0]==">"): 
            print("Inequality in IC50!!!\n")
         else:
            x.append(float(line)) 
 
    return x

def get_split(X,index_1,index_2,max_size):
   X1 = [X[i] for i in index_1]
   X1 = np.reshape(X1,[len(X1),max_size])
   X2 = [X[i] for i in index_2]
   X2 = np.reshape(X2,[len(X2),max_size])
   return X1,X2

def train_dev_split_inter(data,index_train,index_dev,max_size1,max_size2):
    print(len(data))
    print((max_size1,max_size2))
    data_train = [data[i] for i in index_train]
    data_train = np.asarray(data_train)
    data_train = np.reshape(data_train,[len(data_train),max_size1,max_size2])
    data_dev = [data[i] for i in index_dev]
    data_dev = np.reshape(data_dev,[len(data_dev),max_size1,max_size2])

    return data_train,data_dev


def  train_dev_split(train_protein,train_prot_contacts,train_compound_ver,train_compound_adj,train_IC50,train_group,train_prot_inter,train_prot_inter_exist,train_prot_length, train_comp_length,dev_perc,protein_max_size,comp_max_size,group_size,num_group,batch_size):
    num_whole= len(train_IC50)
    num_train = math.ceil(num_whole*(1-dev_perc))
    num_dev = num_whole - num_train

    index_total = range(0,num_whole)
    index_dev = sorted(random.sample(index_total,num_dev))
    remain = list(set(index_total)^set(index_dev))
    index_train = sorted(random.sample(remain,num_train))

    compound_train_ver = [train_compound_ver[i] for i in index_train]
    compound_train_ver = np.reshape(compound_train_ver,[len(compound_train_ver),comp_max_size,feature_num])
    compound_dev_ver = [train_compound_ver[i] for i in index_dev]
    compound_dev_ver = np.reshape(compound_dev_ver,[len(compound_dev_ver),comp_max_size,feature_num])


    compound_train_adj = [train_compound_adj[i] for i in index_train]
    compound_train_adj = np.reshape(compound_train_adj,[len(compound_train_adj),comp_max_size,comp_max_size])
    compound_dev_adj = [train_compound_adj[i] for i in index_dev]
    compound_dev_adj = np.reshape(compound_dev_adj,[len(compound_dev_adj),comp_max_size,comp_max_size])


    IC50_train = [train_IC50[i] for i in index_train]
    IC50_train = np.reshape(IC50_train,[len(IC50_train),1])
    IC50_dev = [train_IC50[i] for i in index_dev]
    IC50_dev = np.reshape(IC50_dev,[len(IC50_dev),1])

    protein_train = [train_protein[i] for i in index_train]
    protein_train = np.reshape(protein_train,[len(protein_train),num_group,group_size])
    protein_dev = [train_protein[i] for i in index_dev]
    protein_dev = np.reshape(protein_dev,[len(protein_dev),num_group,group_size])


    prot_train_length = [train_prot_length[i] for i in index_train]
    prot_train_length = np.reshape(prot_train_length,[len(prot_train_length),1])
    prot_dev_length = [train_prot_length[i] for i in index_dev]
    prot_dev_length = np.reshape(prot_dev_length,[len(prot_dev_length),1])

    comp_train_length = [train_comp_length[i] for i in index_train]
    comp_train_length = np.reshape(comp_train_length,[len(comp_train_length),1])
    comp_dev_length = [train_comp_length[i] for i in index_dev]
    comp_dev_length = np.reshape(comp_dev_length,[len(comp_dev_length),1])

    group_train = [train_group[i] for i in index_train]
    group_dev = [train_group[i] for i in index_dev]


    prot_train_contacts = np.asarray([train_prot_contacts[i] for i in index_train])
    prot_train_contacts = np.reshape(prot_train_contacts,[len(prot_train_contacts),protein_max_size,protein_max_size])
    prot_dev_contacts = np.asarray([train_prot_contacts[i] for i in index_dev])
    prot_dev_contacts = np.reshape(prot_dev_contacts,[len(prot_dev_contacts),protein_max_size,protein_max_size])



    prot_train_inter,prot_dev_inter = train_dev_split_inter(train_prot_inter,index_train,index_dev,full_prot_size,comp_max_size)
    prot_train_inter_exist,prot_dev_inter_exist = get_split(train_prot_inter_exist,index_train,index_dev,1)


    return compound_train_ver, compound_dev_ver,compound_train_adj, compound_dev_adj, IC50_train, IC50_dev, protein_train, protein_dev,prot_train_contacts,prot_dev_contacts,group_train, group_dev, prot_train_inter,prot_dev_inter, prot_train_inter_exist,prot_dev_inter_exist,prot_train_length,prot_dev_length,comp_train_length,comp_dev_length

def read_interaction(path,MAX_size1,MAX_size2):
  g = open(path,"r")
  inter = []
  for line in g:
      line = line.strip().split(' ')
      temp = [float(x) for x in line]
      if len(temp) < MAX_size2:
         pad = [0] * (MAX_size2 - len(temp))
         temp = temp+pad

      inter.append(temp)

  cur_len = len(inter)
  for i in range(MAX_size1 - cur_len):
        temp = [0]*MAX_size2
        inter.append(temp)


  g.close()
  return inter


def create_dummy_interaction(MAX_size1,MAX_size2):
   inter = []
   for i in range(MAX_size1):
       temp = [0]*MAX_size2
       inter.append(temp)

   return inter

def calculate_dict_interaction(protein_unique_inter_path,MAX_size1,MAX_size2,protein_inter_path):
  unique_inter = []
  dict_inter = {}
  count=0
  f = open(protein_unique_inter_path, "r")
  for uin in f:
      uin = uin.strip()
      inter = read_interaction(protein_inter_path+'/'+uin+'.atom',MAX_size1,MAX_size2)
      unique_inter.append(inter)
      dict_inter[uin] = count
      count += 1

  f.close()
  return dict_inter,unique_inter

def count_matrix_size(file_path):
  count = 0
  with open(file_path) as f:
    for line in f:
      line = line.strip().replace(' ','')
      count += line.count('1')
  return count


def read_protein_interactions(source_path,matrix_path,dict_inter,unique_inter,MAX_size1,MAX_size2):
  dataset_inter = []
  dataset_inter_exist = []
  mycount=0
  '''
  min_len = float('inf')
  with open(source_path) as f:
    for line in f:
      line = line.strip()
      if line != "":
        m_size = count_matrix_size(matrix_path + '/' + line + '.atom')
        if m_size < min_len:
          min_len = m_size
  '''
  with tf.gfile.GFile(source_path, mode="r") as source_file:
      source = source_file.readline()
      counter = 0
      while source:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()

        source = source.strip()
        if source != "":
            x = unique_inter[dict_inter[source]]
            m_size = count_matrix_size(matrix_path + '/' + source + '.atom')
            e = 1.0 / m_size
        else:
            x = create_dummy_interaction(MAX_size1,MAX_size2)
            e = 0

        dataset_inter.append(x)
        dataset_inter_exist.append(e)

        mycount=mycount+1
        source = source_file.readline()
  #dataset_inter = np.reshape(dataset_inter,(mycount,MAX_size1,MAX_size2))
  dataset_inter = np.asarray(dataset_inter)
  dataset_inter_exist = np.asarray(dataset_inter_exist)
  return dataset_inter,dataset_inter_exist

def read_protein_group(source_path):
  group = []
  with tf.gfile.GFile(source_path, mode="r") as source_file:
      source = source_file.readline()
      counter = 0
      while source:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source = source.strip()
        if source != "":
          source = source.split(',')
          group.append(source)
        source = source_file.readline()
  return group

def read_length(source_path):
  length = []
  with tf.gfile.GFile(source_path, mode="r") as source_file:
      source = source_file.readline()
      counter = 0
      while source:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        l = len(source.strip())
        length.append(l)
        source = source_file.readline()
  return length

def read_atom_length(source_path):
  length = []
  with tf.gfile.GFile(source_path, mode="r") as source_file:
      source = source_file.readline()
      counter = 0
      while source:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        mol = Chem.MolFromSmiles(source)
        Chem.SanitizeMol(mol)
        l = len(mol.GetAtoms())
        length.append(l)
        source = source_file.readline()
  return length


def convert_to_full_matrix(small_matrix, group_data, prot_max_size, full_prot_size, comp_max_size):
  start_index = 0
  full_matrix = np.zeros((full_prot_size,comp_max_size))
  for i in range(len(group_data)):
      for j in range(comp_max_size):
          end_index = int(group_data[i])
          value = small_matrix[i][j]
          while start_index <= end_index:
              full_matrix[start_index][j] = value
              start_index += 1
  return full_matrix

def trim_matrix(pred_label, length_prot, length_comp):
    for i in range(len(pred_label)):
        if i < length_prot:
            for j in range(len(pred_label[0])):
                if j >= length_comp:
                    pred_label[i][j] = 0
        else:
            pred_label[i] = [0] * len(pred_label[i])
    return pred_label


def convert_joint_attn_matrix(alpha,joint_attn,num_group):
    for i in range(num_group):
        temp1 = np.array([alpha[i,:]])
        temp2 = np.array([joint_attn[i,:]])
        if i == 0:
            result = np.dot(temp1.T,temp2)
        else:
            result = np.concatenate((result,np.dot(temp1.T,temp2)),axis=0)
    return result

def create_fused_matrix(max_size1,max_size2):
    mat_fused = []
    temp = np.zeros((max_size1,))
    temp[0] = 1
    temp[1] = -1
    #temp = np.tile(x,[max_size1,1])
    mat_fused.append(temp)
    for i in range(1,max_size1):
        temp = np.roll(temp,shift=1,axis=0)
        mat_fused.append(temp)

    mat_fused = np.asarray(mat_fused)
    mat_fused = tf.convert_to_tensor(mat_fused, np.float32)
    return mat_fused

def read_contactmap(path,MAX_size):
  g = open(path,"r")
  contacts = []
  for line in g:
      line = line.strip().split(' ')
      temp = [float(x) for x in line]
      if len(temp) < MAX_size:
         pad = [0] * (MAX_size - len(temp))
         temp = temp+pad

      contacts.append(temp)

  cur_len = len(contacts)
  for i in range(MAX_size - cur_len):
        temp = [0]*MAX_size
        contacts.append(temp)


  g.close()
  return contacts

def calculate_dict_contact(protein_unique_contact_path,MAX_size,protein_contact_path):

  unique_contacts = []
  dict_contacts = {}
  count=0
  f = open(protein_unique_contact_path, "r")
  for uin in f:
      uin = uin.strip()
      contact = read_contactmap(protein_contact_path+'/'+uin+'_contactmap.txt',MAX_size)
      unique_contacts.append(contact)
      dict_contacts[uin] = count
      count += 1

  f.close()
  return dict_contacts,unique_contacts


def read_protein_contacts(source_path,dict_contacts,unique_contacts):
  dataset_contacts = []
  mycount=0
  with tf.gfile.GFile(source_path, mode="r") as source_file:
      source = source_file.readline()
      counter = 0
      while source:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()

        source = source.strip()
        unique_contacts[dict_contacts[source]]
        dataset_contacts.append(unique_contacts[dict_contacts[source]])

        mycount=mycount+1
        source = source_file.readline()
  dataset_contacts = np.asarray(dataset_contacts)
  return dataset_contacts

def plot_ref(true_list, pred_list, x_range,name):
  x_tp = []
  y_tp = []
  x_fp = []
  y_fp = []
  x_true = []
  y_true = []
  for term in pred_list:
    if term in true_list:
      x_tp.append(term[0])
      y_tp.append(term[1])
    else:
      x_fp.append(term[0])
      y_fp.append(term[1])
  for term in true_list:
    x_true.append(term[0])
    y_true.append(term[1])
  plt.plot(x_true, y_true, 'o', color='gray',markersize=1.5,label='Ground Truth')
  plt.plot(x_fp, y_fp, 'o', color='red',markersize=1.5,label='Mismatch')
  plt.plot(x_tp, y_tp, 'o', color='black',markersize=1.5,label='Match')
  plt.legend(bbox_to_anchor=(0,1,1,0.1), loc="lower left",mode="expand",ncol=3)
  plt.xlabel('Protein', fontsize=10)
  plt.ylabel('Compound', fontsize=10)
  plt.xlim(-5, x_range + 5)
  #plt.title(name.split('/')[-1], fontsize=15, y=1.10)
  #plt.show()
  plt.savefig(name, format="svg",bbox_inches='tight')
  plt.clf()

def plot_matrix(true_matrix, pred_matrix, name, k=2):
  num = 0
  top_k = {}
  pred_list = []
  true_list = []
  x_range = len(true_matrix)
  for i in range(len(true_matrix)):
    for j in range(len(true_matrix[0])):
      if true_matrix[i][j] == 1:
        num += 1
        true_list.append([i,j])

  num = num * k
  for i in range(len(true_matrix)):
    for j in range(len(true_matrix[0])):
      if len(top_k) < num:
        top_k[pred_matrix[i][j]] = [[i,j]]
      else:
        current_min = min(top_k.keys())
        if pred_matrix[i][j] > current_min:
          del top_k[current_min]
          top_k[pred_matrix[i][j]] = [[i,j]]
          current_min = min(top_k.keys())
        elif pred_matrix[i][j] == current_min:
          top_k[current_min].append([i,j])
  for i in top_k.values():
    for j in i:
      pred_list.append(j)
  plot_ref(true_list, pred_list, x_range,name + '_2x')

  for k in [0.01, 0.02, 0.05]:
    top_k = {}
    pred_list = []
    num = int(len(true_matrix) * len(true_matrix[0]) * k)
    for i in range(len(true_matrix)):
      for j in range(len(true_matrix[0])):
        if len(top_k) < num:
          top_k[pred_matrix[i][j]] = [[i,j]]
        else:
          current_min = min(top_k.keys())
          if pred_matrix[i][j] > current_min:
            del top_k[current_min]
            top_k[pred_matrix[i][j]] = [[i,j]]
            current_min = min(top_k.keys())
          elif pred_matrix[i][j] == current_min:
            top_k[current_min].append([i,j])
    for i in top_k.values():
      for j in i:
        pred_list.append(j)
    plot_ref(true_list, pred_list, x_range,name + '_' + str(k))

def cal_index_diff(current_index,last_index):
    if re.search(r'[a-zA-Z]',current_index):
        current_num = int(current_index[:-1])
    else:
        current_num = int(current_index)
    if re.search(r'[a-zA-Z]',last_index):
        last_num = int(last_index[:-1])
    else:
        last_num = int(last_index)
    if abs(current_num - last_num) <= 1:
        return 1
    else:
        return abs(current_num - last_num)



def align_seq(seq_p,seq_p_index, seq_u):
    # return the start index of seq_p in seq_u. if seq_p starts from the beginning, it will be 1.
    pdb_uniprot_mapping = {}
    if len(seq_p) <= 30:
        print(seq_p)
        return "error"
    i = 0
    while i <= len(seq_p)-18:
        sub_seq = seq_p[i:i+18]
        index = seq_u.find(sub_seq)
        if index != -1:
            current_index_pdb = i
            last_index_pdb = i
            current_index_uniprot = index
            pdb_uniprot_mapping[seq_p_index[current_index_pdb]] = str(current_index_uniprot)  # index of string, start from 0
            while current_index_pdb > 0:
                current_index_pdb -= 1
                diff = cal_index_diff(seq_p_index[current_index_pdb], seq_p_index[last_index_pdb])
                if diff >= 2:
                    sub_seq = seq_p[current_index_pdb - 9:current_index_pdb + 1]
                    index_temp = seq_u[:current_index_uniprot].find(sub_seq)
                    if index_temp != -1:
                        current_index_uniprot = index_temp + 9
                    else:
                        index_temp = seq_u.find(sub_seq)
                        if index_temp != -1:
                            current_index_uniprot = index_temp + 9
                        else:
                            current_index_uniprot -= 1
                else:
                    current_index_uniprot -= diff
                pdb_uniprot_mapping[seq_p_index[current_index_pdb]] = str(current_index_uniprot)
                last_index_pdb = current_index_pdb
            current_index_pdb = i
            last_index_pdb = i
            current_index_uniprot = index
            while current_index_pdb < len(seq_p) - 1:
                current_index_pdb += 1
                diff = cal_index_diff(seq_p_index[current_index_pdb], seq_p_index[last_index_pdb])
                if diff >= 2:
                    try:
                        sub_seq = seq_p[current_index_pdb:current_index_pdb + 10]
                    except:
                        sub_seq = seq_p[current_index_pdb:]
                    index_temp = seq_u[current_index_uniprot + 1:].find(sub_seq)
                    if index_temp != -1:
                        current_index_uniprot += index_temp + 1
                    else:
                        index_temp = seq_u.find(sub_seq)
                        if index_temp != -1:
                            current_index_uniprot = index_temp
                        else:
                            current_index_uniprot += 1
                else:
                    current_index_uniprot += diff
                pdb_uniprot_mapping[seq_p_index[current_index_pdb]] = str(current_index_uniprot)
                last_index_pdb = current_index_pdb
            return pdb_uniprot_mapping
        i+=1
    print(seq_p,seq_p_index, seq_u)
    return "alignment error"

def pdb_start_index(pdbfile, chain):
    with open(pdbfile) as f:
        for line in f:
            if line[0:4] == "ATOM" and line[21] == chain:
                long_aa = line[17:20]
                aa = pdb2single[long_aa]
                index = int(line[22:26].strip())
                return aa,index

def comp_index(index1,index2):
    index1 = index1[0]
    index2 = index2[0]
    if re.search(r'[A-Z]+',index1):
        v1 = int(index1[:-1]) + 0.1 * (ord(index1[-1]) - ord('A') + 1)
    else:
        v1 = int(index1)
    if re.search(r'[A-Z]+',index2):
        v2 = int(index2[:-1]) + 0.1 * (ord(index2[-1]) - ord('A') + 1)
    else:
        v2 = int(index2)
    return v1 - v2
def pdb2seq(pdbfile):
    seq = {}
    string = ''
    last_index = 0
    last_chain = ''
    seq_index = []
    with open(pdbfile) as f:
        for line in f:
            if line[0:4] == "ATOM" or line[0:6] == "HETATM" and line[17:20] in pdb2single:
                current_chain = line[21]
                long_aa = line[17:20]
                aa = pdb2single[long_aa]
                index = line[22:27].strip()
                if last_chain == current_chain:
                    if index != last_index:
                        string += aa
                        seq_index.append(index)
                        last_index = index
                else:
                    if last_chain != '':
                        if last_chain not in seq:
                            index_seq = sorted(zip(seq_index,list(string)),key = cmp_to_key(comp_index))
                            string = ''.join(x[1] for x in index_seq)
                            seq_index = [x[0] for x in index_seq]
                            seq[last_chain] = [string,seq_index]
                    string = aa
                    seq_index = [index]
                    last_index = index
                    last_chain = current_chain
    if current_chain not in seq:
        index_seq = sorted(zip(seq_index,list(string)),key = cmp_to_key(comp_index))
        string = ''.join(x[1] for x in index_seq)
        seq_index = [x[0] for x in index_seq]
        seq[current_chain] = [string,seq_index]
    return seq

def interaction_chain(interaction_file):
    m = 0
    n = 0
    with open(interaction_file) as f:
        for line in f:
            old_line = line
            line = re.split(r'[ ]+', line.strip())
            if re.match(r'[0-9]+\.', line[0]) and m == 0:
                chain = line[5]
                comp_chain = old_line[59:62].strip()
                ligend_res_index = int(line[10])
                m = 1
            elif line[0] == 'PDB' and line[1] == 'code:':
                pid = line[2]
                n = 1
    if m == 1 and n == 1:
        return chain, pid, ligend_res_index,comp_chain
    else:
        print ("Error on:",interaction_file)
        return "error"

def count_valid_lines(interaction_file, chain):
    count = 0
    with open(interaction_file) as f:
        for line in f:
            start = re.split(r'[ ]+', line)[0]
            if re.match(r'[0-9.]+', start) and line[29] == chain:
                count += 1
    return count

def binding_site(top_k):
    binding = []
    for i in top_k:
        binding.append(i[1][0])
    return binding

def get_icode(current_index):
    if re.search(r'[a-zA-Z]',current_index):
        icode = current_index[-1]
        index = int(current_index[:-1])
    else:
        icode = ' '
        index = int(current_index)
    return index, icode

def binding_cluster(pdb_file, pdb_length, het, chain, comp_chain, ligend_res_index, pdb_uniprot_mapping):
    cutoff_list = [0,4,6,8,10]
    cluster = []
    current_index_set = set()
    ppb = CaPPBuilder()
    parser = PDBParser()
    structure = parser.get_structure('pdb', pdb_file)
    while len(het) < 3:
      het = ' ' + het
    if comp_chain == '':
      comp_chain = ' '
    for i in range(len(cutoff_list)-1):
        cutoff_lower_bound = cutoff_list[i]
        cutoff_upper_bound = cutoff_list[i+1]
        cluster_set = set()
        for atom_ligand in structure[0][comp_chain][('H_' + het, ligend_res_index,' ')]:
            # i_shift = i + shift
            for current_index in pdb_uniprot_mapping:
                #current_index = j + start_index_pdb
                current_index_u = int(pdb_uniprot_mapping[current_index])
                current_index, icode = get_icode(current_index)
                if current_index in structure[0][chain] and current_index_u not in current_index_set:
                    for atom_protein in structure[0][chain][' ', current_index, icode]:
                        if atom_ligand - atom_protein <= cutoff_upper_bound and atom_ligand - atom_protein > cutoff_lower_bound:
                            cluster_set.add(current_index_u)
                            current_index_set.add(current_index_u)
                            break
        cluster.append(cluster_set)
    return cluster


def cal_cluster(name, interaction_dir, pdb_dir, pred_matrix, seq_uniprot):
    top_50_p = []
    save_index = {}
    k = 50
    for i in range(len(pred_matrix)):
        for j in range(len(pred_matrix[0])):
            if len(save_index) < k:
                save_index[pred_matrix[i][j]] = [i,j]
            else:
                current_min = min(save_index.keys())
                if float(pred_matrix[i][j]) > current_min:
                    del save_index[current_min]
                    save_index[float(pred_matrix[i][j])] = [i,j]
    top_50 = sorted(save_index.items(),reverse=True)[:k]
    top_50 = binding_site(top_50)

    interact_chain, pid, ligend_res_index,comp_chain = interaction_chain(interaction_dir + name)
    seq_pdb_list = pdb2seq(pdb_dir + pid + ".pdb")
    seq_pdb = seq_pdb_list[interact_chain][0]
    seq_pdb_index = seq_pdb_list[interact_chain][1]
    pdb_uniprot_mapping = align_seq(seq_pdb,seq_pdb_index, seq_uniprot)

    het = name.split('_')[1].split('-')[0]
    true_binding_cluster = binding_cluster(pdb_dir + pid + ".pdb",  len(seq_pdb), het, interact_chain,comp_chain, ligend_res_index, pdb_uniprot_mapping)
    for binding in true_binding_cluster:
        top_50_p_temp = 0
        for i in top_50:
            if i in binding:
                top_50_p_temp += 1.0
        top_50_p.append(top_50_p_temp)
    top_50_p.append(len(top_50) - sum(top_50_p))
    top_50_p = np.array(top_50_p)/len(top_50)
    return top_50_p/sum(top_50_p)
def get_seq(seq_num):
    seq_num = np.reshape(seq_num,(group_size*num_group,))
    result_seq = ''
    #print("get_seq:",seq_num)
    for i in seq_num:
        if i != 0:
            result_seq += index_aa[i]
        else:
            return result_seq
    return result_seq








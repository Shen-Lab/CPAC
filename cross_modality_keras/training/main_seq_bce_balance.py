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

import warnings
# from Bio import BiopythonWarning
# warnings.simplefilter('ignore', BiopythonWarning)
# from Bio.PDB import *

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

import pdb





################################################ Hyper-parameters #################################
l1_fused_binding = 0.0001 # float(sys.argv[1]) #0.0001
l2_group_binding = 0.001 # float(sys.argv[2]) #0.001
l1_binding = 0.01 # float(sys.argv[3]) #0.01
# lambda_interaction = 1000 # float(sys.argv[4]) # 1000
lambda_interaction = 1000 # float(sys.argv[1]) # 1000
seed = 0 # int(sys.argv[2]) # 0


from utils import *
from nets import *


np.random.seed(seed)
random.seed(seed)
tf.set_random_seed(seed)


## preparing data 
"""
fused_matrix = create_fused_matrix(prot_max_size,comp_max_size)
prot_dict_contacts, prot_uniq_contacts = calculate_dict_contact("../data/merged_data/contact_matrix/uniq_contact.txt",prot_max_size,"../data/merged_data/contact_matrix")
print('finish loading data 1')


train_prot_length = read_length(data_dir+"/train_seq")
train_group = read_protein_group(data_dir+"/train_group")
train_protein = prepare_data(data_dir,data_dir+"/train_seq",vocab_size_protein,vocab_protein,prot_max_size,group_size)
train_IC50 = read_labels(data_dir+"/train_kd")
train_comp_length = read_atom_length(data_dir+"/train_smi")
train_compound_ver,train_compound_adj = read_graph(data_dir+"/train_smi",comp_max_size,dict_atom,dict_atom_hal_don,dict_atom_hybridization)
train_prot_contacts = read_protein_contacts(data_dir+"/train_uid",prot_dict_contacts,prot_uniq_contacts)
print('finish loading data 2')

test_prot_length = read_length(data_dir+"/test_seq")
test_group = read_protein_group(data_dir+"/test_group")
test_protein = prepare_data(data_dir,data_dir+"/test_seq",vocab_size_protein,vocab_protein,prot_max_size,group_size)
test_label = read_labels(data_dir+"/test_kd")
test_comp_length = read_atom_length(data_dir+"/test_smi")
test_compound_ver,test_compound_adj = read_graph(data_dir+"/test_smi",comp_max_size,dict_atom,dict_atom_hal_don,dict_atom_hybridization)
test_prot_contacts = read_protein_contacts(data_dir+"/test_uid",prot_dict_contacts,prot_uniq_contacts)
print('finish loading data 3')


'''
Gen_prot_length = read_length(data_dir+"/genera_seq")
Gen_group = read_protein_group(data_dir+"/genera_group")
Gen_protein = prepare_data(data_dir,data_dir+"/genera_seq",vocab_size_protein,vocab_protein,prot_max_size,group_size)
Gen_label = read_labels(data_dir+"/genera_kd")
Gen_comp_length = read_atom_length(data_dir+"/genera_smi")
Gen_compound_ver,Gen_compound_adj = read_graph(data_dir+"/genera_smi",comp_max_size,dict_atom,dict_atom_hyd,dict_atom_hal_acc,dict_atom_hal_don,dict_atom_hybridization)
Gen_prot_contacts = read_protein_contacts(data_dir+"/genera_uid",prot_dict_contacts,prot_uniq_contacts)
'''

protein_uniq_prot_length = read_length(data_dir+"/protein_uniq_seq")
protein_uniq_group = read_protein_group(data_dir+"/protein_uniq_group")
protein_uniq_protein = prepare_data(data_dir,data_dir+"/protein_uniq_seq",vocab_size_protein,vocab_protein,prot_max_size,group_size)
protein_uniq_label = read_labels(data_dir+"/protein_uniq_kd")
protein_uniq_comp_length = read_atom_length(data_dir+"/protein_uniq_smi")
protein_uniq_compound_ver,protein_uniq_compound_adj = read_graph(data_dir+"/protein_uniq_smi",comp_max_size,dict_atom,dict_atom_hal_don,dict_atom_hybridization)
protein_uniq_prot_contacts = read_protein_contacts(data_dir+"/protein_uniq_uid",prot_dict_contacts,prot_uniq_contacts)
print('finish loading data 4')

compound_uniq_prot_length = read_length(data_dir+"/compound_uniq_seq")
compound_uniq_group = read_protein_group(data_dir+"/compound_uniq_group")
compound_uniq_protein = prepare_data(data_dir,data_dir+"/compound_uniq_seq",vocab_size_protein,vocab_protein,prot_max_size,group_size)
compound_uniq_label = read_labels(data_dir+"/compound_uniq_kd")
compound_uniq_comp_length = read_atom_length(data_dir+"/compound_uniq_smi")
compound_uniq_compound_ver,compound_uniq_compound_adj = read_graph(data_dir+"/compound_uniq_smi",comp_max_size,dict_atom,dict_atom_hal_don,dict_atom_hybridization)
compound_uniq_prot_contacts = read_protein_contacts(data_dir+"/compound_uniq_uid",prot_dict_contacts,prot_uniq_contacts)
print('finish loading data 5')

double_uniq_prot_length = read_length(data_dir+"/double_uniq_seq")
double_uniq_group = read_protein_group(data_dir+"/double_uniq_group")
double_uniq_protein = prepare_data(data_dir,data_dir+"/double_uniq_seq",vocab_size_protein,vocab_protein,prot_max_size,group_size)
double_uniq_label = read_labels(data_dir+"/double_uniq_kd")
double_uniq_comp_length = read_atom_length(data_dir+"/double_uniq_smi")
double_uniq_compound_ver,double_uniq_compound_adj = read_graph(data_dir+"/double_uniq_smi",comp_max_size,dict_atom,dict_atom_hal_don,dict_atom_hybridization)
double_uniq_prot_contacts = read_protein_contacts(data_dir+"/double_uniq_uid",prot_dict_contacts,prot_uniq_contacts)
print('finish loading data 6')



##inter data

prot_dict_inter_whole, prot_uniq_inter_whole = calculate_dict_interaction("../data/merged_data/whole_matrix/uniq_interactions.txt",full_prot_size,comp_max_size,"../data/merged_data/whole_matrix")
train_prot_inter,train_prot_inter_exist = read_protein_interactions(data_dir+"/train_inter","../data/merged_data/whole_matrix",prot_dict_inter_whole,prot_uniq_inter_whole,full_prot_size,comp_max_size)
test_prot_inter,test_prot_inter_exist = read_protein_interactions(data_dir+"/test_inter","../data/merged_data/whole_matrix",prot_dict_inter_whole,prot_uniq_inter_whole,full_prot_size,comp_max_size)
protein_uniq_prot_inter,protein_uniq_prot_inter_exist = read_protein_interactions(data_dir+"/protein_uniq_inter","../data/merged_data/whole_matrix",prot_dict_inter_whole,prot_uniq_inter_whole,full_prot_size,comp_max_size)
compound_uniq_prot_inter,compound_uniq_prot_inter_exist = read_protein_interactions(data_dir+"/compound_uniq_inter","../data/merged_data/whole_matrix",prot_dict_inter_whole,prot_uniq_inter_whole,full_prot_size,comp_max_size)
double_uniq_prot_inter,double_uniq_prot_inter_exist = read_protein_interactions(data_dir+"/double_uniq_inter","../data/merged_data/whole_matrix",prot_dict_inter_whole,prot_uniq_inter_whole,full_prot_size,comp_max_size)
print('finish loading data 7')

#Gen_prot_inter,Gen_prot_inter_exist = read_protein_interactions(data_dir+"/genera_inter",prot_dict_inter_whole,prot_uniq_inter_whole,full_prot_size,comp_max_size)


## separating train,dev, test data
compound_train_ver, compound_dev_ver,compound_train_adj, compound_dev_adj, IC50_train, IC50_dev, protein_train, protein_dev,prot_train_contacts,prot_dev_contacts, group_train, group_dev, prot_train_inter,prot_dev_inter, prot_train_inter_exist,prot_dev_inter_exist,prot_train_length,prot_dev_length,comp_train_length,comp_dev_length = train_dev_split(train_protein,train_prot_contacts,train_compound_ver,train_compound_adj,train_IC50,train_group,train_prot_inter,train_prot_inter_exist,train_prot_length,train_comp_length, dev_perc,prot_max_size,comp_max_size,group_size,num_group,batch_size)
print('finish loading data 8')



"""

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


# load processed data
data_processed_dir = '/scratch/user/yuning.you/proj/DeepRelations/HierRNN_GCN_constraint_sup/data_processed/'
protein_train = np.load(data_processed_dir+'protein_train.npy')
compound_train_ver = np.load(data_processed_dir+'compound_train_ver.npy')
compound_train_adj = np.load(data_processed_dir+'compound_train_adj.npy')
prot_train_contacts = np.load(data_processed_dir+'prot_train_contacts.npy')
prot_train_inter = np.load(data_processed_dir+'prot_train_inter.npy')
prot_train_inter_exist = np.load(data_processed_dir+'prot_train_inter_exist.npy')
IC50_train = np.load(data_processed_dir+'IC50_train.npy')

# pdb.set_trace()

fused_matrix = tf.convert_to_tensor(np.load(data_processed_dir+'fused_matrix.npy'))

protein_dev = np.load(data_processed_dir+'protein_dev.npy')
compound_dev_ver = np.load(data_processed_dir+'compound_dev_ver.npy')
compound_dev_adj = np.load(data_processed_dir+'compound_dev_adj.npy')
prot_dev_contacts = np.load(data_processed_dir+'prot_dev_contacts.npy')
prot_dev_inter = np.load(data_processed_dir+'prot_dev_inter.npy')
prot_dev_inter_exist = np.load(data_processed_dir+'prot_dev_inter_exist.npy')
IC50_dev = np.load(data_processed_dir+'IC50_dev.npy')
'''

dp = './data_processed/sar/JAK2/'
protein_dev = np.load(dp+'protein.npy')
prot_dev_contacts = np.load(dp+'protein_adj.npy')
compound_dev_ver = np.load(dp+'compound.npy')
compound_dev_adj = np.load(dp+'compound_adj.npy')
IC50_dev = np.load(dp+'labels.npy')
prot_dev_inter = np.load(data_processed_dir+'prot_dev_inter.npy')
prot_dev_inter_exist = np.load(data_processed_dir+'prot_dev_inter_exist.npy')
'''


protein_test = np.load(data_processed_dir+'protein_test.npy')
compound_test_ver = np.load(data_processed_dir+'compound_test_ver.npy')
compound_test_adj = np.load(data_processed_dir+'compound_test_adj.npy')
prot_test_contacts = np.load(data_processed_dir+'prot_test_contacts.npy')
prot_test_inter = np.load(data_processed_dir+'prot_test_inter.npy')
prot_test_inter_exist = np.load(data_processed_dir+'prot_test_inter_exist.npy')
IC50_test = np.load(data_processed_dir+'IC50_test.npy')





### model
## RNN for protein
prot_inter = Input(shape=(prot_max_size,comp_max_size),name='prot_inter')
prot_inter_exist = Input(shape=(1,),name='prot_inter_exist')

prot_contacts = Input(shape=(prot_max_size,prot_max_size),name='prot_contacts')
prot_data = Input(shape=(num_group,group_size))
amino_embd = TimeDistributed(Embedding(input_dim = vocab_size_protein, output_dim = GRU_size_prot, input_length=group_size))(prot_data)

gru_1 = GRU(units=GRU_size_prot,return_sequences=True)
amino_lstm = TimeDistributed(gru_1)(amino_embd)
amino_lstm = TimeDistributed(Reshape((group_size,GRU_size_prot)))(amino_lstm)


###
amino_lstm_1 = Reshape((40, 25, 256))(amino_lstm)
gru_11 = GRU(units=GRU_size_prot,return_sequences=True)
amino_lstm_1 = TimeDistributed(gru_11)(amino_lstm_1)
###


alphas = TimeDistributed(Sep_attn_alphas(output_dim=GRU_size_prot,length=group_size),name='alphas')(amino_lstm)
prot_encoder = Sep_attn_beta(output_dim=GRU_size_prot,length=num_group)([amino_lstm,alphas])
print("alphas:",alphas.shape)
print("prot_encoder:",prot_encoder.shape)

gru_2 = GRU(units=GRU_size_prot,return_sequences=True,name='after_GRU')
prot_lstm = gru_2(prot_encoder)

prot_lstm = Reshape((num_group,GRU_size_prot))(prot_lstm)


## RNN for drug
drug_data_ver = Input(shape=(comp_max_size,feature_num))
drug_data_adj = Input(shape=(comp_max_size, comp_max_size))
drug_embd = drug_data_ver
drug_embd = graph_layer(output_dim=GRU_size_drug,length=comp_max_size,input_dim=feature_num)([drug_data_adj,drug_embd])
drug_embd = graph_layer(output_dim=GRU_size_drug,length=comp_max_size,input_dim=GRU_size_drug)([drug_data_adj,drug_embd])
drug_embd = graph_layer(output_dim=GRU_size_drug,length=comp_max_size,input_dim=GRU_size_drug)([drug_data_adj,drug_embd])
drug_embd = Reshape((comp_max_size,GRU_size_drug))(drug_embd)

# joint attention
joint_attn = joint_attn(max_size1=num_group,shape1=GRU_size_prot,
              max_size2=comp_max_size,shape2=GRU_size_drug,name='joint_attn')([prot_lstm,drug_embd])
joint_attn_1 = joint_attn_new(max_size1=num_group,shape1=GRU_size_prot,
              max_size2=comp_max_size,shape2=GRU_size_drug,name='joint_attn_new')([amino_lstm_1,drug_embd])
print("joint_attn:",joint_attn.shape)
Attn = joint_vectors(dim=256)([prot_lstm,drug_embd,joint_attn])
reg_FSGL  = FSGL_constraint_new(mylambda_group=l2_group_binding,mylambda_l1=l1_binding,fused_matrix=fused_matrix,mylambda_fused=l1_fused_binding,prot_max_size=prot_max_size,comp_max_size=comp_max_size)([prot_contacts,joint_attn_1])
# reg_inter = interaction_penalty_new(batch=batch_size,mylambda=lambda_interaction)([prot_inter,prot_inter_exist,joint_attn_1])
reg_inter = interaction_penalty_bce_balance(batch=batch_size,mylambda=lambda_interaction)([prot_inter,prot_inter_exist,joint_attn_1])

# conv + fc
conv_1 = Conv1D(filters=64,kernel_size=4,strides=2,padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(0.001))(Attn)
conv_1 = LeakyReLU(alpha=0.1)(conv_1)
pool_1 = MaxPooling1D(pool_size=4)(conv_1)
final = Reshape((64*32,))(pool_1)
## merging
#merging = concatenate([prot_final,drug_final],axis=1)
fc_1 = Dense(units=600,kernel_initializer='glorot_uniform')(final)
fc_1 = add_fake(num=600)([fc_1,reg_FSGL])
fc_1 = add_fake(num=600)([fc_1,reg_inter])
fc_1 = LeakyReLU(alpha=0.1)(fc_1)
drop_2 = Dropout(rate=0.8)(fc_1)
fc_2 = Dense(units=300,kernel_initializer='glorot_uniform')(drop_2)
fc_2 = LeakyReLU(alpha=0.1)(fc_2)
drop_3 = Dropout(rate=0.8)(fc_2)
linear = Dense(units=1,activation="linear",kernel_initializer='glorot_uniform')(drop_3)
model = Model(inputs=[prot_data,drug_data_ver,drug_data_adj,prot_contacts,prot_inter,prot_inter_exist],outputs=[linear])
optimizer = Adam(0.001, clipvalue=5)

# pdb.set_trace()

# print(model.count_params())
# assert False

if lambda_interaction == 0:
  para_num= str(int(math.log10(l1_fused_binding) * -1)) + str(int(math.log10(l2_group_binding) * -1))+ str(int(math.log10(l1_binding) * -1)) + "00"
elif lambda_interaction <= 1:
  para_num = str(int(math.log10(l1_fused_binding) * -1)) + str(int(math.log10(l2_group_binding) * -1))+ str(int(math.log10(l1_binding) * -1)) + str(int(math.log10(lambda_interaction) * -1))
else:
  para_num = str(int(math.log10(l1_fused_binding) * -1)) + str(int(math.log10(l2_group_binding) * -1))+ str(int(math.log10(l1_binding) * -1)) + str(int(lambda_interaction))
para_num += '_lr3_coldstart_newcorrectnor' + dataset + "_newembed"

# filepath="./weights_hrnn_" + str(seed) + "/weights.best.hdf5_hrnn_2dattn_" + str(lambda_interaction) + '_{epoch:02d}'
# filepath="./weights_hrnn_" + str(seed) + "/weights.best.hdf5_hrnn_2dattn_" + str(lambda_interaction) + '_176'
filepath = './weights/deepmodality_seq_bce_balance.h5'

model.compile(loss=penalized_loss(reg_FSGL,reg_inter),
              optimizer=optimizer)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [checkpoint,tensorboard]


import scipy
# Training.
#model.load_weights("weights.best.hdf5_434")
model.fit([protein_train,compound_train_ver,compound_train_adj,prot_train_contacts,prot_train_inter,prot_train_inter_exist], IC50_train,
          batch_size=batch_size,
          epochs=200,
          verbose=1,
          validation_data=([protein_dev,compound_dev_ver,compound_dev_adj,prot_dev_contacts,prot_dev_inter,prot_dev_inter_exist], IC50_dev),
          callbacks=callbacks_list)

## saving

model.load_weights(filepath)
print()
print("error on dev")
size = 64
length_dev = len(protein_dev)
print(length_dev)
num_bins = math.ceil(length_dev/size)
y_pred = model.predict([protein_dev,compound_dev_ver,compound_dev_adj,prot_dev_contacts,prot_dev_inter,prot_dev_inter_exist])
y_pred = np.asarray(y_pred)
er=0
error = []
for i in range(length_dev):
  er += (y_pred[i]-IC50_dev[i])**2
  error.append(abs(y_pred[i]-IC50_dev[i]))
mse = er/length_dev
print('mse', mse)
print('sqrt mse', math.sqrt(mse))
print("std:",np.std(error))

print('pearson', scipy.stats.pearsonr(y_pred, IC50_dev))

print(scipy.stats.kendalltau(y_pred, IC50_dev))
print(scipy.stats.spearmanr(y_pred, IC50_dev))
# assert False

# results = sm.OLS(y_pred,sm.add_constant(IC50_dev)).fit()
# print("R = {}".format(math.sqrt(results.rsquared)))
# print(results.summary())


print()
print("error on Test")
"""
test_protein = np.reshape(test_protein,[len(test_protein),num_group,group_size])
test_compound_adj = np.reshape(test_compound_adj,[len(test_compound_adj),comp_max_size,comp_max_size])
test_compound_ver = np.reshape(test_compound_ver,[len(test_compound_ver),comp_max_size,feature_num])
"""
y_pred = model.predict([protein_test,compound_test_ver,compound_test_adj,prot_test_contacts,prot_test_inter,prot_test_inter_exist])
print(protein_test.shape)
length_test, _, _ = protein_test.shape
er=0
error = []
for i in range(length_test):
  er += (y_pred[i]-IC50_test[i])**2
  error.append(abs(y_pred[i]-IC50_test[i]))
with open("./logs/test_error_" + para_num,'w+') as w:
  w.write('\n'.join([str(x[0]) for x in error]))
mse = er/length_test
print('mse', mse)
print('sqrt mse', math.sqrt(mse))
print("std:",np.std(error))

print('pearson', scipy.stats.pearsonr(y_pred[:,0], IC50_test))

# results = sm.OLS(y_pred,sm.add_constant(test_label)).fit()
# print("R = {}".format(math.sqrt(results.rsquared)))
# print(results.summary())
np.savetxt("./logs/test_pred_affinity_" + para_num,np.asarray(y_pred))

'''
print("error on Gen")
Gen_protein = np.reshape(Gen_protein,[len(Gen_protein),num_group,group_size])
Gen_compound_adj = np.reshape(Gen_compound_adj,[len(Gen_compound_adj),comp_max_size,comp_max_size])
Gen_compound_ver = np.reshape(Gen_compound_ver,[len(Gen_compound_ver),comp_max_size,feature_num])


y_pred = model.predict([Gen_protein,Gen_compound_ver,Gen_compound_adj,Gen_prot_contacts,Gen_prot_inter,Gen_prot_inter_exist])
length_Gen = len(Gen_protein)
er=0
error = []

for i in range(length_Gen):
  er += (y_pred[i]-Gen_label[i])**2
  error.append(abs(y_pred[i]-Gen_label[i]))
print("std:",np.std(error))

mse = er/length_Gen
print(mse)
print(math.sqrt(mse))

results = sm.OLS(y_pred,sm.add_constant(Gen_label)).fit()
print("R = {}".format(math.sqrt(results.rsquared)))
print(results.summary())
'''

print()
print("error on train")
"""
protein_train = np.reshape(protein_train,[len(protein_train),num_group,group_size])
compound_train_adj = np.reshape(compound_train_adj,[len(compound_train_adj),comp_max_size,comp_max_size])
compound_train_ver = np.reshape(compound_train_ver,[len(compound_train_ver),comp_max_size,feature_num])
"""
y_pred = model.predict([protein_train,compound_train_ver,compound_train_adj,prot_train_contacts,prot_train_inter,prot_train_inter_exist])
length_train = len(protein_train)
er=0
error = []
for i in range(length_train):
  er += (y_pred[i]-IC50_train[i])**2
  error.append(abs(y_pred[i]-IC50_train[i]))
mse = er/length_train
print('mse', mse)
print('sqrt mse', math.sqrt(mse))
print("std:",np.std(error))
# results = sm.OLS(y_pred,sm.add_constant(IC50_train)).fit()
# print("R = {}".format(math.sqrt(results.rsquared)))
# print(results.summary())
np.savetxt("./logs/train_pred_affinity_" + para_num,np.asarray(y_pred))






print()
print("error on protein unique")
protein_uniq_protein = np.load(data_processed_dir+'protein_uniq_protein.npy')
protein_uniq_compound_ver = np.load(data_processed_dir+'protein_uniq_compound_ver.npy')
protein_uniq_compound_adj = np.load(data_processed_dir+'protein_uniq_compound_adj.npy')
protein_uniq_prot_contacts = np.load(data_processed_dir+'protein_uniq_prot_contacts.npy')
protein_uniq_prot_inter = np.load(data_processed_dir+'protein_uniq_prot_inter.npy')
protein_uniq_prot_inter_exist = np.load(data_processed_dir+'protein_uniq_prot_inter_exist.npy')
protein_uniq_label = np.load(data_processed_dir+'protein_uniq_label.npy')
"""
protein_uniq_protein = np.reshape(protein_uniq_protein,[len(protein_uniq_protein),num_group,group_size])
protein_uniq_compound_adj = np.reshape(protein_uniq_compound_adj,[len(protein_uniq_compound_adj),comp_max_size,comp_max_size])
protein_uniq_compound_ver = np.reshape(protein_uniq_compound_ver,[len(protein_uniq_compound_ver),comp_max_size,feature_num])
"""
y_pred = model.predict([protein_uniq_protein,protein_uniq_compound_ver,protein_uniq_compound_adj,protein_uniq_prot_contacts,protein_uniq_prot_inter,protein_uniq_prot_inter_exist])
length_protein_uniq = len(protein_uniq_protein)
er=0
error = []
for i in range(length_protein_uniq):
  er += (y_pred[i]-protein_uniq_label[i])**2
  error.append(abs(y_pred[i]-protein_uniq_label[i]))
with open("./logs/protein_unique_error_" + para_num,'w+') as w:
  w.write('\n'.join([str(x[0]) for x in error]))
mse = er/length_protein_uniq
print('mse', mse)
print('sqrt mse', math.sqrt(mse))
print("std:",np.std(error))
# results = sm.OLS(y_pred,sm.add_constant(protein_uniq_label)).fit()
# print("R = {}".format(math.sqrt(results.rsquared)))
# print(results.summary())
np.savetxt("./logs/prot_uniq_pred_affinity_" + para_num,np.asarray(y_pred))


print('pearson', scipy.stats.pearsonr(y_pred[:,0], protein_uniq_label))



print()
print("error on compound unique")
compound_uniq_protein = np.load(data_processed_dir+'compound_uniq_protein.npy')
compound_uniq_compound_ver = np.load(data_processed_dir+'compound_uniq_compound_ver.npy')
compound_uniq_compound_adj = np.load(data_processed_dir+'compound_uniq_compound_adj.npy')
compound_uniq_prot_contacts = np.load(data_processed_dir+'compound_uniq_prot_contacts.npy')
compound_uniq_prot_inter = np.load(data_processed_dir+'compound_uniq_prot_inter.npy')
compound_uniq_prot_inter_exist = np.load(data_processed_dir+'compound_uniq_prot_inter_exist.npy')
compound_uniq_label = np.load(data_processed_dir+'compound_uniq_label.npy')
"""
compound_uniq_protein = np.reshape(compound_uniq_protein,[len(compound_uniq_protein),num_group,group_size])
compound_uniq_compound_adj = np.reshape(compound_uniq_compound_adj,[len(compound_uniq_compound_adj),comp_max_size,comp_max_size])
compound_uniq_compound_ver = np.reshape(compound_uniq_compound_ver,[len(compound_uniq_compound_ver),comp_max_size,feature_num])
"""
y_pred = model.predict([compound_uniq_protein,compound_uniq_compound_ver,compound_uniq_compound_adj,compound_uniq_prot_contacts,compound_uniq_prot_inter,compound_uniq_prot_inter_exist])
length_compound_uniq = len(compound_uniq_protein)
er=0
error = []
for i in range(length_compound_uniq):
  er += (y_pred[i]-compound_uniq_label[i])**2
  error.append(abs(y_pred[i]-compound_uniq_label[i]))
with open("./logs/compound_unique_error_" + para_num,'w+') as w:
  w.write('\n'.join([str(x[0]) for x in error]))
mse = er/length_compound_uniq
print('mse', mse)
print('sqrt mse', math.sqrt(mse))
print("std:",np.std(error))



print('pearson', scipy.stats.pearsonr(y_pred[:,0], compound_uniq_label))


"""
#################################
# print(type(y_pred), type(sm.add_constant(compound_uniq_label)))
print(y_pred.shape, sm.add_constant(compound_uniq_label).shape)
#################################
results = sm.OLS(y_pred,sm.add_constant(compound_uniq_label)).fit()
print("R = {}".format(math.sqrt(results.rsquared)))
print(results.summary())
"""
np.savetxt("./logs/comp_uniq_pred_affinity_" + para_num,np.asarray(y_pred))


print()
print("error on double unique")
double_uniq_protein = np.load(data_processed_dir+'double_uniq_protein.npy')
double_uniq_compound_ver = np.load(data_processed_dir+'double_uniq_compound_ver.npy')
double_uniq_compound_adj = np.load(data_processed_dir+'double_uniq_compound_adj.npy')
double_uniq_prot_contacts = np.load(data_processed_dir+'double_uniq_prot_contacts.npy')
double_uniq_prot_inter = np.load(data_processed_dir+'double_uniq_prot_inter.npy')
double_uniq_prot_inter_exist = np.load(data_processed_dir+'double_uniq_prot_inter_exist.npy')
double_uniq_label = np.load(data_processed_dir+'double_uniq_label.npy')
"""
double_uniq_protein = np.reshape(double_uniq_protein,[len(double_uniq_protein),num_group,group_size])
double_uniq_compound_adj = np.reshape(double_uniq_compound_adj,[len(double_uniq_compound_adj),comp_max_size,comp_max_size])
double_uniq_compound_ver = np.reshape(double_uniq_compound_ver,[len(double_uniq_compound_ver),comp_max_size,feature_num])
"""
y_pred = model.predict([double_uniq_protein,double_uniq_compound_ver,double_uniq_compound_adj,double_uniq_prot_contacts,double_uniq_prot_inter,double_uniq_prot_inter_exist])
length_double_uniq = len(double_uniq_protein)
er=0
error = []
for i in range(length_double_uniq):
  er += (y_pred[i]-double_uniq_label[i])**2
  error.append(abs(y_pred[i]-double_uniq_label[i]))
with open("./logs/double_unique_error_" + para_num,'w+') as w:
  w.write('\n'.join([str(x[0]) for x in error]))
mse = er/length_double_uniq
print('mse', mse)
print('sqrt mse', math.sqrt(mse))
print("std:",np.std(error))


print('pearson', scipy.stats.pearsonr(y_pred[:,0], double_uniq_label))


"""
#################################
# print(type(y_pred), type(sm.add_constant(double_uniq_label)))
print(y_pred.shape, sm.add_constant(double_uniq_label).shape)
#################################
results = sm.OLS(y_pred,sm.add_constant(double_uniq_label)).fit()
print("R = {}".format(math.sqrt(results.rsquared)))
print(results.summary())
"""
np.savetxt("./logs/double_uniq_pred_affinity_" + para_num,np.asarray(y_pred))



######### Interpretability for dev
print()
print("dev")
pred_dir = "pred_inter"
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
f = open(data_dir+"/train_inter", "r")
names = []
for n in f:
    n = n.strip()
    names.append(n)
f.close()
'''
print('ALPHAS-----------------------------------------------------------')
layer_name = ['alphas'] #list of layers
mylayers_output =  [model.get_layer(l).output for l in layer_name]
inp = model.input
inp.append(K.learning_phase())
functor = K.function(inp, mylayers_output )   # evaluation function
prot_dev_inter_exist = np.reshape(prot_dev_inter_exist,(len(prot_dev_inter_exist),1))
#test_comp_adj = np.reshape(test_comp_adj,(len(test_comp_adj),comp_max_size,comp_max_size))
layer_outs = functor([protein_dev,compound_dev_ver,compound_dev_adj, 1.])  # predict for given inputs

print(np.asarray(layer_outs).shape)
print('after_GRU-----------------------------------------------------------')
layer_name = ['after_GRU'] #list of layers
mylayers_output =  [model.get_layer(l).output for l in layer_name]
inp = model.input
inp.append(K.learning_phase())
functor = K.function(inp, mylayers_output )   # evaluation function
prot_dev_inter_exist = np.reshape(prot_dev_inter_exist,(len(prot_dev_inter_exist),1))
#test_comp_adj = np.reshape(test_comp_adj,(len(test_comp_adj),comp_max_size,comp_max_size))
layer_outs = functor([protein_dev,compound_dev_ver,compound_dev_adj, 1.])  # predict for given inputs

print(np.asarray(layer_outs).shape)

print('------------------------------------------------------------')
'''
layer_name = ['alphas','joint_attn_new'] #list of layers
mylayers_output =  [model.get_layer(l).output for l in layer_name]
inp = model.input
inp.append(K.learning_phase())
functor = K.function(inp, mylayers_output )   # evaluation function

prot_dev_inter_exist = np.reshape(prot_dev_inter_exist,(len(prot_dev_inter_exist),1))
#test_comp_adj = np.reshape(test_comp_adj,(len(test_comp_adj),comp_max_size,comp_max_size))
layer_outs = functor([protein_dev,compound_dev_ver,compound_dev_adj, prot_dev_contacts,prot_dev_inter,prot_dev_inter_exist,1.])  # predict for given inputs
AP_1 = []
AUC_1 = []
AP_1_margin = []
AUC_1_margin = []


prot_dev_length = np.load('./data_processed/prot_dev_length.npy')
comp_dev_length = np.load('./data_processed/comp_dev_length.npy')


count=0
for i in range(len(prot_dev_inter_exist)):
    if prot_dev_inter_exist[i] != 0:
        count += 1
        length_prot = prot_dev_length[i][0]
        length_comp = comp_dev_length[i][0]
        true_label = np.asarray(prot_dev_inter[i])[:length_prot,:length_comp]
        # true_label = np.reshape(true_label_cut,(length_prot*length_comp))

        pred_label_1 = convert_joint_attn_matrix(np.asarray(layer_outs[0][i]),np.asarray(layer_outs[1][i]),num_group)[:length_prot,:length_comp]
        # pdb.set_trace() ###

        if i % 100 == 0:
            np.save('attn_viz/label_'+str(i)+'_hrnn.npy', true_label)
            np.save('attn_viz/pred_'+str(i)+'_hrnn.npy', pred_label_1)

# assert False






count=0
for i in range(len(prot_dev_inter_exist)):
    if prot_dev_inter_exist[i] != 0:
        count += 1
        length_prot = prot_dev_length[i][0]
        length_comp = comp_dev_length[i][0]
        true_label_cut = np.asarray(prot_dev_inter[i])[:length_prot,:length_comp]
        true_label = np.reshape(true_label_cut,(length_prot*length_comp))

        full_matrix = np.asarray(layer_outs[1][i])[:length_prot,:length_comp]
        # pdb.set_trace()

        pred_label_1 = np.reshape(full_matrix,(length_prot*length_comp))

        average_precision_whole = average_precision_score(true_label, pred_label_1)
        AP_1.append(average_precision_whole)

        fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label_1)
        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC_1.append(roc_auc_whole)

        true_label = np.amax(true_label_cut,axis=1)
        pred_label_1 = np.amax(full_matrix,axis=1)

        average_precision_whole = average_precision_score(true_label, pred_label_1)
        AP_1_margin.append(average_precision_whole)

        fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label_1)
        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC_1_margin.append(roc_auc_whole)

        np.savetxt(pred_dir+"/"+names[i]+"_joint_attn.txt", full_matrix, delimiter=',')

print("interaction")
mean = np.mean(AUC_1)
std = np.std(AUC_1)
print("AUC = "+str(mean) + " +/- "+str(std))
mean = np.mean(AP_1)
std = np.std(AP_1)
print("AP = "+str(mean) + " +/- "+str(std))

print("binding")
mean = np.mean(AUC_1_margin)
std = np.std(AUC_1_margin)
print("AUC = "+str(mean) + " +/- "+str(std))
mean = np.mean(AP_1_margin)
std = np.std(AP_1_margin)
print("AP = "+str(mean) + " +/- "+str(std))
print(count)
np.savetxt("./logs/AUC_deepaffinity_dev.txt",AUC_1,delimiter=',')
np.savetxt("./logs/AP_deepaffinity_dev.txt",AP_1,delimiter=',')


######### Interpretability for Test
print()
print("Test")
pred_dir = "pred_inter"
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

f = open(data_dir+"/test_inter", "r")
names = []
for n in f:
    n = n.strip()
    names.append(n)

f.close()
layer_name = ['alphas','joint_attn_new'] #list of layers
mylayers_output =  [model.get_layer(l).output for l in layer_name]
inp = model.input
inp.append(K.learning_phase())
functor = K.function(inp, mylayers_output )   # evaluation function

prot_test_inter_exist = np.reshape(prot_test_inter_exist,(len(prot_test_inter_exist),1))
#test_comp_adj = np.reshape(test_comp_adj,(len(test_comp_adj),comp_max_size,comp_max_size))
layer_outs = functor([protein_test,compound_test_ver,compound_test_adj,prot_test_contacts,prot_test_inter,prot_test_inter_exist, 1.])  # predict for given inputs
AP_1 = []
AUC_1 = []
AP_1_margin = []
AUC_1_margin = []
AP_1_name = []
cluster_result = np.zeros(5)


prot_test_length = np.load('./data_processed/prot_test_length.npy')
comp_test_length = np.load('./data_processed/comp_test_length.npy')


count=0
for i in range(len(prot_test_inter_exist)):
    if prot_test_inter_exist[i] != 0:
        count += 1
        length_prot = prot_test_length[i]
        length_comp = comp_test_length[i]
        true_label_cut = np.asarray(prot_test_inter[i])[:length_prot,:length_comp]
        true_label = np.reshape(true_label_cut,(length_prot*length_comp))

        full_matrix = np.asarray(layer_outs[1][i])[:length_prot,:length_comp]
        pred_label_1 = np.reshape(full_matrix,(length_prot*length_comp))
        # cluster = cal_cluster(names[i], interaction_dir, pdb_dir, full_matrix, get_seq(protein_test[i]))
        # cluster_result += cluster
        average_precision_whole = average_precision_score(true_label, pred_label_1)
        AP_1.append(average_precision_whole)
        AP_1_name.append(names[i])
        fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label_1)
        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC_1.append(roc_auc_whole)

        true_label = np.amax(true_label_cut,axis=1)
        pred_label_1 = np.amax(full_matrix,axis=1)

        average_precision_whole = average_precision_score(true_label, pred_label_1)
        AP_1_margin.append(average_precision_whole)

        fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label_1)
        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC_1_margin.append(roc_auc_whole)

        np.savetxt(pred_dir+"/"+names[i]+"_joint_attn.txt", full_matrix, delimiter=',')

# print("[4,8,12,16,20,+inf]:\n",cluster_result/count)
with open("./logs/test_AUPRC",'w+') as f:
    temp_str = ''
    for temp_s in AP_1_name:
        temp_str += str(temp_s) + ' '
    f.write(temp_str + '\n')
    temp_str = ''
    for temp_s in AP_1:
        temp_str += str(temp_s) + ' '
    f.write(temp_str)

print("interaction")
mean = np.mean(AUC_1)
std = np.std(AUC_1)
print("AUC = "+str(mean) + " +/- "+str(std))
mean = np.mean(AP_1)
std = np.std(AP_1)
print("AP = "+str(mean) + " +/- "+str(std))

print("binding")
mean = np.mean(AUC_1_margin)
std = np.std(AUC_1_margin)
print("AUC = "+str(mean) + " +/- "+str(std))
mean = np.mean(AP_1_margin)
std = np.std(AP_1_margin)
print("AP = "+str(mean) + " +/- "+str(std))

print(count)
np.savetxt("./logs/AUC_deepaffinity_test_" + para_num + ".txt",AUC_1,delimiter=',')
np.savetxt("./logs/AP_deepaffinity_test_" + para_num + ".txt",AP_1,delimiter=',')



######### Interpretability for protein unique
print()
print("protein unique")
pred_dir = "pred_inter"
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

f = open(data_dir+"/protein_uniq_inter", "r")
names = []
for n in f:
    n = n.strip()
    names.append(n)

f.close()
layer_name = ['alphas','joint_attn_new'] #list of layers
mylayers_output =  [model.get_layer(l).output for l in layer_name]
inp = model.input
inp.append(K.learning_phase())
functor = K.function(inp, mylayers_output )   # evaluation function

protein_uniq_prot_inter_exist = np.reshape(protein_uniq_prot_inter_exist,(len(protein_uniq_prot_inter_exist),1))
#protein_uniq_comp_adj = np.reshape(protein_uniq_comp_adj,(len(protein_uniq_comp_adj),comp_max_size,comp_max_size))
layer_outs = functor([protein_uniq_protein,protein_uniq_compound_ver,protein_uniq_compound_adj,protein_uniq_prot_contacts,protein_uniq_prot_inter,protein_uniq_prot_inter_exist, 1.])  # predict for given inputs
AP_1 = []
AUC_1 = []
AP_1_margin = []
AUC_1_margin = []
AP_1_name = []
cluster_result = np.zeros(5)


protein_uniq_prot_length = np.load('./data_processed/protein_uniq_prot_length.npy')
protein_uniq_comp_length = np.load('./data_processed/protein_uniq_comp_length.npy')

count=0
for i in range(len(protein_uniq_prot_inter_exist)):
    if protein_uniq_prot_inter_exist[i] != 0:
        count += 1
        length_prot = protein_uniq_prot_length[i]
        length_comp = protein_uniq_comp_length[i]

        true_label_cut = np.asarray(protein_uniq_prot_inter[i])[:length_prot,:length_comp]
        true_label = np.reshape(true_label_cut,(length_prot*length_comp))

        full_matrix = np.asarray(layer_outs[1][i])[:length_prot,:length_comp]
        pred_label_1 = np.reshape(full_matrix,(length_prot*length_comp))
        # cluster = cal_cluster(names[i], interaction_dir, pdb_dir, full_matrix, get_seq(protein_uniq_protein[i]))
        # cluster_result += cluster
        average_precision_whole = average_precision_score(true_label, pred_label_1)
        AP_1.append(average_precision_whole)
        AP_1_name.append(names[i])
        fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label_1)
        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC_1.append(roc_auc_whole)

        true_label = np.amax(true_label_cut,axis=1)
        pred_label_1 = np.amax(full_matrix,axis=1)

        average_precision_whole = average_precision_score(true_label, pred_label_1)
        AP_1_margin.append(average_precision_whole)

        fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label_1)
        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC_1_margin.append(roc_auc_whole)


        np.savetxt(pred_dir+"/"+names[i]+"_joint_attn.txt", full_matrix, delimiter=',')
# print("[4,8,12,16,20,+inf]:\n",cluster_result/count)
with open("./logs/protein_unique_AUPRC",'w+') as f:
    temp_str = ''
    for temp_s in AP_1_name:
        temp_str += str(temp_s) + ' '
    f.write(temp_str + '\n')
    temp_str = ''
    for temp_s in AP_1:
        temp_str += str(temp_s) + ' '
    f.write(temp_str)

print("interaction")
mean = np.mean(AUC_1)
std = np.std(AUC_1)
print("AUC = "+str(mean) + " +/- "+str(std))
mean = np.mean(AP_1)
std = np.std(AP_1)
print("AP = "+str(mean) + " +/- "+str(std))

print("binding")
mean = np.mean(AUC_1_margin)
std = np.std(AUC_1_margin)
print("AUC = "+str(mean) + " +/- "+str(std))
mean = np.mean(AP_1_margin)
std = np.std(AP_1_margin)
print("AP = "+str(mean) + " +/- "+str(std))

print(count)
np.savetxt("./logs/AUC_deepaffinity_protein_uniq_" + para_num + ".txt",AUC_1,delimiter=',')
np.savetxt("./logs/AP_deepaffinity_protein_uniq_" + para_num + ".txt",AP_1,delimiter=',')


######### Interpretability for compound unique
print()
print("compound unique")
pred_dir = "pred_inter"
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

f = open(data_dir+"/compound_uniq_inter", "r")
names = []
for n in f:
    n = n.strip()
    names.append(n)

f.close()
layer_name = ['alphas','joint_attn_new'] #list of layers
mylayers_output =  [model.get_layer(l).output for l in layer_name]
inp = model.input
inp.append(K.learning_phase())
functor = K.function(inp, mylayers_output )   # evaluation function

compound_uniq_prot_inter_exist = np.reshape(compound_uniq_prot_inter_exist,(len(compound_uniq_prot_inter_exist),1))
#compound_uniq_comp_adj = np.reshape(compound_uniq_comp_adj,(len(compound_uniq_comp_adj),comp_max_size,comp_max_size))
layer_outs = functor([compound_uniq_protein,compound_uniq_compound_ver,compound_uniq_compound_adj, compound_uniq_prot_contacts,compound_uniq_prot_inter,compound_uniq_prot_inter_exist,1.])  # predict for given inputs
AP_1 = []
AUC_1 = []
AP_1_margin = []
AUC_1_margin = []
AP_1_name = []
cluster_result = np.zeros(5)

compound_uniq_prot_length = np.load('./data_processed/compound_uniq_prot_length.npy')
compound_uniq_comp_length = np.load('./data_processed/compound_uniq_comp_length.npy')


count=0
for i in range(len(compound_uniq_prot_inter_exist)):
    if compound_uniq_prot_inter_exist[i] != 0:
        count += 1
        length_prot = compound_uniq_prot_length[i]
        length_comp = compound_uniq_comp_length[i]

        true_label_cut = np.asarray(compound_uniq_prot_inter[i])[:length_prot,:length_comp]
        true_label = np.reshape(true_label_cut,(length_prot*length_comp))
        full_matrix = np.asarray(layer_outs[1][i])[:length_prot,:length_comp]
        pred_label_1 = np.reshape(full_matrix,(length_prot*length_comp))
        # cluster = cal_cluster(names[i], interaction_dir, pdb_dir, full_matrix, get_seq(compound_uniq_protein[i]))
        # cluster_result += cluster
        average_precision_whole = average_precision_score(true_label, pred_label_1)
        AP_1.append(average_precision_whole)
        AP_1_name.append(names[i])
        fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label_1)
        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC_1.append(roc_auc_whole)

        true_label = np.amax(true_label_cut,axis=1)
        pred_label_1 = np.amax(full_matrix,axis=1)

        average_precision_whole = average_precision_score(true_label, pred_label_1)
        AP_1_margin.append(average_precision_whole)

        fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label_1)
        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC_1_margin.append(roc_auc_whole)


        np.savetxt(pred_dir+"/"+names[i]+"_joint_attn.txt", full_matrix, delimiter=',')
# print("[4,8,12,16,20,+inf]:\n",cluster_result/count)
with open("./logs/compound_unique_AUPRC",'w+') as f:
    temp_str = ''
    for temp_s in AP_1_name:
        temp_str += str(temp_s) + ' '
    f.write(temp_str + '\n')
    temp_str = ''
    for temp_s in AP_1:
        temp_str += str(temp_s) + ' '
    f.write(temp_str)
print("interaction")
mean = np.mean(AUC_1)
std = np.std(AUC_1)
print("AUC = "+str(mean) + " +/- "+str(std))
mean = np.mean(AP_1)
std = np.std(AP_1)
print("AP = "+str(mean) + " +/- "+str(std))

print("binding")
mean = np.mean(AUC_1_margin)
std = np.std(AUC_1_margin)
print("AUC = "+str(mean) + " +/- "+str(std))
mean = np.mean(AP_1_margin)
std = np.std(AP_1_margin)
print("AP = "+str(mean) + " +/- "+str(std))

print(count)
np.savetxt("./logs/AUC_deepaffinity_compound_uniq_" + para_num + ".txt",AUC_1,delimiter=',')
np.savetxt("./logs/AP_deepaffinity_compound_uniq_" + para_num + ".txt",AP_1,delimiter=',')


######### Interpretability for double unique
print()
print("double unique")
pred_dir = "pred_inter"
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

f = open(data_dir+"/double_uniq_inter", "r")
names = []
for n in f:
    n = n.strip()
    names.append(n)

f.close()
layer_name = ['alphas','joint_attn_new'] #list of layers
mylayers_output =  [model.get_layer(l).output for l in layer_name]
inp = model.input
inp.append(K.learning_phase())
functor = K.function(inp, mylayers_output )   # evaluation function

double_uniq_prot_inter_exist = np.reshape(double_uniq_prot_inter_exist,(len(double_uniq_prot_inter_exist),1))
#double_uniq_comp_adj = np.reshape(double_uniq_comp_adj,(len(double_uniq_comp_adj),comp_max_size,comp_max_size))
layer_outs = functor([double_uniq_protein,double_uniq_compound_ver,double_uniq_compound_adj, double_uniq_prot_contacts,double_uniq_prot_inter,double_uniq_prot_inter_exist, 1.])  # predict for given inputs
AP_1 = []
AUC_1 = []
AP_1_margin = []
AUC_1_margin = []
AP_1_name = []
cluster_result = np.zeros(5)

double_uniq_prot_length = np.load('./data_processed/double_uniq_prot_length.npy')
double_uniq_comp_length = np.load('./data_processed/double_uniq_comp_length.npy')

count=0
for i in range(len(double_uniq_prot_inter_exist)):
    if double_uniq_prot_inter_exist[i] != 0:
        count += 1
        length_prot = double_uniq_prot_length[i]
        length_comp = double_uniq_comp_length[i]

        true_label_cut = np.asarray(double_uniq_prot_inter[i])[:length_prot,:length_comp]
        true_label = np.reshape(true_label_cut,(length_prot*length_comp))

        full_matrix = np.asarray(layer_outs[1][i])[:length_prot,:length_comp]
        pred_label_1 = np.reshape(full_matrix,(length_prot*length_comp))
        # cluster = cal_cluster(names[i], interaction_dir, pdb_dir, full_matrix, get_seq(double_uniq_protein[i]))
        # cluster_result += cluster
        average_precision_whole = average_precision_score(true_label, pred_label_1)
        AP_1.append(average_precision_whole)
        AP_1_name.append(names[i])

        fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label_1)
        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC_1.append(roc_auc_whole)

        true_label = np.amax(true_label_cut,axis=1)
        pred_label_1 = np.amax(full_matrix,axis=1)

        average_precision_whole = average_precision_score(true_label, pred_label_1)
        AP_1_margin.append(average_precision_whole)

        fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label_1)
        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC_1_margin.append(roc_auc_whole)


        np.savetxt(pred_dir+"/"+names[i]+"_joint_attn.txt", full_matrix, delimiter=',')
# print("[4,8,12,16,20,+inf]:\n",cluster_result/count)
print("interaction")
mean = np.mean(AUC_1)
std = np.std(AUC_1)
print("AUC = "+str(mean) + " +/- "+str(std))
mean = np.mean(AP_1)
std = np.std(AP_1)
print("AP = "+str(mean) + " +/- "+str(std))

print("binding")
mean = np.mean(AUC_1_margin)
std = np.std(AUC_1_margin)
print("AUC = "+str(mean) + " +/- "+str(std))
mean = np.mean(AP_1_margin)
std = np.std(AP_1_margin)
print("AP = "+str(mean) + " +/- "+str(std))

print(count)
np.savetxt("./logs/AUC_deepaffinity_double_uniq_" + para_num + ".txt",AUC_1,delimiter=',')
np.savetxt("./logs/AP_deepaffinity_double_uniq_" + para_num + ".txt",AP_1,delimiter=',')



######### Interpretability for train
print()
print("train")
pred_dir = "pred_inter"
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

f = open(data_dir+"/train_inter", "r")
names = []
for n in f:
    n = n.strip()
    names.append(n)


assert False

f.close()
layer_name = ['alphas','joint_attn'] #list of layers
mylayers_output =  [model.get_layer(l).output for l in layer_name]
inp = model.input
inp.append(K.learning_phase())
functor = K.function(inp, mylayers_output )   # evaluation function

prot_train_inter_exist = np.reshape(prot_train_inter_exist,(len(prot_train_inter_exist),1))
#test_comp_adj = np.reshape(test_comp_adj,(len(test_comp_adj),comp_max_size,comp_max_size))
num_bins = math.ceil(len(protein_train)/batch_size)
for i in range(num_bins):
  if i == 0:
    layer_outs = functor([protein_train[:batch_size],compound_train_ver[:batch_size],compound_train_adj[:batch_size],prot_train_contacts[:batch_size],prot_train_inter[:batch_size],prot_train_inter_exist[:batch_size], 1.])
  elif i < num_bins-1:
    temp = functor([protein_train[(i*size):((i+1)*size)],compound_train_ver[(i*size):((i+1)*size)],compound_train_adj[(i*size):((i+1)*size)],prot_train_contacts[(i*size):((i+1)*size)],prot_train_inter[(i*size):((i+1)*size)],prot_train_inter_exist[(i*size):((i+1)*size)], 1.])
    layer_outs[0] = np.concatenate((layer_outs[0],temp[0]), axis=0)
    layer_outs[1] = np.concatenate((layer_outs[1],temp[1]), axis=0)

  else:
    temp = functor([protein_train[(i*size):],compound_train_ver[(i*size):],compound_train_adj[(i*size):],prot_train_contacts[(i*size):],prot_train_inter[(i*size):],prot_train_inter_exist[(i*size):], 1.])
    layer_outs[0] = np.concatenate((layer_outs[0],temp[0]), axis=0)
    layer_outs[1] = np.concatenate((layer_outs[1],temp[1]), axis=0)



#layer_outs = functor([protein_train,compound_train_ver,compound_train_adj, prot_train_contacts,prot_train_inter,prot_train_inter_exist,1.])  # predict for given inputs
AP_1 = []
AUC_1 = []
AP_1_margin = []
AUC_1_margin = []

prot_train_length = np.load('./data_processed/prot_train_length.npy')
comp_train_length = np.load('./data_processed/comp_train_length.npy')

count=0
for i in range(len(prot_train_inter_exist)):
    if prot_train_inter_exist[i] != 0:
        count += 1
        length_prot = prot_train_length[i][0]
        length_comp = comp_train_length[i][0]

        true_label_cut = np.asarray(prot_train_inter[i])[:length_prot,:length_comp]
        true_label = np.reshape(true_label_cut,(length_prot*length_comp))

        # convert small (layer_outs[0][i]) to big matrix
        full_matrix = convert_joint_attn_matrix(np.asarray(layer_outs[0][i]),np.asarray(layer_outs[1][i]),num_group)[:length_prot,:length_comp]
        pred_label_1 = np.reshape(full_matrix,(length_prot*length_comp))

        average_precision_whole = average_precision_score(true_label, pred_label_1)
        AP_1.append(average_precision_whole)

        fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label_1)
        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC_1.append(roc_auc_whole)

        true_label = np.amax(true_label_cut,axis=1)
        pred_label_1 = np.amax(full_matrix,axis=1)

        average_precision_whole = average_precision_score(true_label, pred_label_1)
        AP_1_margin.append(average_precision_whole)

        fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label_1)
        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC_1_margin.append(roc_auc_whole)



        np.savetxt(pred_dir+"/"+names[i]+"_joint_attn.txt", full_matrix, delimiter=',')

print("interaction")
mean = np.mean(AUC_1)
std = np.std(AUC_1)
print("AUC = "+str(mean) + " +/- "+str(std))
mean = np.mean(AP_1)
std = np.std(AP_1)
print("AP = "+str(mean) + " +/- "+str(std))

print("binding")
mean = np.mean(AUC_1_margin)
std = np.std(AUC_1_margin)
print("AUC = "+str(mean) + " +/- "+str(std))
mean = np.mean(AP_1_margin)
std = np.std(AP_1_margin)
print("AP = "+str(mean) + " +/- "+str(std))


print(count)
np.savetxt("./logs/AUC_deepaffinity_train.txt",AUC_1,delimiter=',')
np.savetxt("./logs/AP_deepaffinity_train.txt",AP_1,delimiter=',')

'''
######### Interpretability for Gen
print("Gen")
pred_dir = "pred_inter"
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

f = open(data_dir+"/genera_inter", "r")
names = []
for n in f:
    n = n.strip()
    names.append(n)

f.close()
layer_name = ['alphas','joint_attn'] #list of layers
mylayers_output =  [model.get_layer(l).output for l in layer_name]
inp = model.input
inp.append(K.learning_phase())
functor = K.function(inp, mylayers_output )   # evaluation function

Gen_prot_inter_exist = np.reshape(Gen_prot_inter_exist,(len(Gen_prot_inter_exist),1))
#test_comp_adj = np.reshape(test_comp_adj,(len(test_comp_adj),comp_max_size,comp_max_size))
layer_outs = functor([Gen_protein,Gen_compound_ver,Gen_compound_adj, Gen_prot_contacts,Gen_prot_inter,Gen_prot_inter_exist,1.])  # predict for given inputs
AP_1 = []
AUC_1 = []
AP_1_margin = []
AUC_1_margin = []


count=0
for i in range(len(Gen_prot_inter_exist)):
    if Gen_prot_inter_exist[i] == 1:
        count += 1
        length_prot = Gen_prot_length[i]
        length_comp = Gen_comp_length[i]
        true_label_cut = np.asarray(Gen_prot_inter[i])[:length_prot,:length_comp]
        true_label = np.reshape(true_label_cut,(length_prot*length_comp))

        # convert small (layer_outs[0][i]) to big matrix
        #full_matrix = convert_to_full_matrix(layer_outs[0][i], Gen_group[i], prot_max_size, full_prot_size, comp_max_size)
        #full_matrix = trim_matrix(full_matrix, length_prot, length_comp)
        #full_matrix = np.asarray(full_matrix)[:length_prot,:length_comp]
        full_matrix = convert_joint_attn_matrix(np.asarray(layer_outs[0][i]),np.asarray(layer_outs[1][i]),num_group)[:length_prot,:length_comp]
        pred_label_1 = np.reshape(full_matrix,(length_prot*length_comp))

        average_precision_whole = average_precision_score(true_label, pred_label_1)
        AP_1.append(average_precision_whole)

        fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label_1)
        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC_1.append(roc_auc_whole)

        true_label = np.amax(true_label_cut,axis=1)
        pred_label_1 = np.amax(full_matrix,axis=1)

        average_precision_whole = average_precision_score(true_label, pred_label_1)
        AP_1_margin.append(average_precision_whole)

        fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label_1)
        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC_1_margin.append(roc_auc_whole)

        np.savetxt(pred_dir+"/"+names[i]+"_joint_attn.txt", layer_outs[0][i], delimiter=',')

print("interaction")
mean = np.mean(AUC_1)
std = np.std(AUC_1)
print("AUC = "+str(mean) + " +/- "+str(std))
mean = np.mean(AP_1)
std = np.std(AP_1)
print("AP = "+str(mean) + " +/- "+str(std))

print("binding")
mean = np.mean(AUC_1_margin)
std = np.std(AUC_1_margin)
print("AUC = "+str(mean) + " +/- "+str(std))
mean = np.mean(AP_1_margin)
std = np.std(AP_1_margin)
print("AP = "+str(mean) + " +/- "+str(std))

print(count)
np.savetxt("AUC_deepaffinity_Gen.txt",AUC_1,delimiter=',')
np.savetxt("AP_deepaffinity_Gen.txt",AP_1,delimiter=',')
'''

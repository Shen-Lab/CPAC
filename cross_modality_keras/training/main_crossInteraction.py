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
from keras.activations import relu, sigmoid
from keras.engine.topology import Layer
from keras.callbacks import ModelCheckpoint,TensorBoard

"""
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem, ChemicalFeatures, rdchem
from rdkit import RDConfig
import rdkit.Chem.rdPartialCharges as rdPartialCharges
"""

from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve,average_precision_score
from functools import cmp_to_key

import pdb




from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)




################################################ Hyper-parameters #################################
l1_fused_binding = 0.001 #  float(sys.argv[1]) #0.0001
l2_group_binding = 0.0001 #  float(sys.argv[2]) #0.001
l1_binding = 0.01 #  float(sys.argv[3]) #0.01
# lambda_interaction = 1000 # float(sys.argv[4]) # 1000
lambda_interaction = 10000 # float(sys.argv[4]) # 1000
seed = 0 # int(sys.argv[5]) # 1000
ww = 0.5
num_gnn_layer = 7
# filepath="./weights_hrnn_gcn_" + str(seed) + "/weights_gcn_newattn_true_" + str(l1_fused_binding) + '_' + str(l2_group_binding) + '_' + str(l1_binding) + '_' + str(lambda_interaction) + '_' +  str(num_gnn_layer) + "_{epoch:02d}"
# filepath="./weights_hrnn_gcn_0" + "/weights_gcn_newattn_true_" + str(l1_fused_binding) + '_' + str(l2_group_binding) + '_' + str(l1_binding) + '_' + str(lambda_interaction) + '_' +  str(num_gnn_layer) + "_105"
# filepath="./weights_hrnn_gcn_crossattn_0" + "/weights_gcn_newattn_true_" + str(l1_fused_binding) + '_' + str(l2_group_binding) + '_' + str(l1_binding) + '_' + str(lambda_interaction) + '_' +  str(num_gnn_layer)

filepath = './weights/deepmodality_crossInteraction.h5'


from utils import *
from nets import *


np.random.seed(seed)
random.seed(seed)
tf.set_random_seed(seed)





# load processed data
data_processed_dir = '../../data/'
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

import scipy
dp = './data_processed/sar/TIE2/'
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



###############################
gnn_hyperparameter_dict = {'num_gnn_layer':num_gnn_layer,
                           'num_dense_layer':2,
                           'drop_out':False,
                           'drop_out_rate':0.5,
                           'skip_connection':True,
                           'num_group':num_group, 'group_size':group_size, 'GRU_size_prot':GRU_size_prot}


class gcn(Layer):
    def __init__(self, hyperparameter_dict={}):
        super(gcn, self).__init__()
        self.hyperparameter_dict = hyperparameter_dict

    def build(self, input_shape):  # Create the state of the layer (weights)
        self.dense_list = [ [Dense(self.hyperparameter_dict['GRU_size_prot'], activation='relu', use_bias=True) for _ in range(self.hyperparameter_dict['num_dense_layer'])] for _ in range(self.hyperparameter_dict['num_gnn_layer'])]
        #? self.dense_final = Dense(self.hyperparameter_dict['GRU_size_prot'], activation='tanh', use_bias=True)
        self.dense_final = Dense(self.hyperparameter_dict['GRU_size_prot'], activation=None, use_bias=True)
        self.drop_out = Dropout(self.hyperparameter_dict['drop_out_rate'], input_shape=(self.hyperparameter_dict['GRU_size_prot'],))

    def call(self, inputs):  # Defines the computation from inputs to outputs
        amino_embd = inputs[0]
        prot_contacts = inputs[1]
        prot_contacts_self_loop = prot_contacts + K.eye(self.hyperparameter_dict['num_group']*self.hyperparameter_dict['group_size'])

        x = Reshape(target_shape=(self.hyperparameter_dict['num_group']*self.hyperparameter_dict['group_size'], self.hyperparameter_dict['GRU_size_prot']))(amino_embd)
        prot_contacts_sum = tf.einsum('bij->bi', prot_contacts_self_loop)
        prot_contacts_norm = tf.einsum('bij,bi->bij', prot_contacts_self_loop, 1/prot_contacts_sum)
        adj = prot_contacts_norm


        # x = self.bn_feat(x)
        for l in range(self.hyperparameter_dict['num_gnn_layer']):
            x0 = x
            x = tf.einsum('bij,bjd->bid', adj, x0)

            for dl in range(self.hyperparameter_dict['num_dense_layer']):
                x = self.dense_list[l][dl](x)

            if self.hyperparameter_dict['drop_out']:
                x = self.drop_out(x)
            if self.hyperparameter_dict['skip_connection']:
                x = x + x0


        x = self.dense_final(x)
        x = Reshape(target_shape=(self.hyperparameter_dict['num_group'], self.hyperparameter_dict['group_size'], self.hyperparameter_dict['GRU_size_prot']))(x)
        return x



class gcn_non_local(Layer):
    def __init__(self, hyperparameter_dict={}):
        super(gcn_non_local, self).__init__()
        self.hyperparameter_dict = hyperparameter_dict

    def build(self, input_shape):  # Create the state of the layer (weights)
        self.dense_list = [ [Dense(self.hyperparameter_dict['GRU_size_prot'], activation='relu', use_bias=True) for _ in range(self.hyperparameter_dict['num_dense_layer'])] for _ in range(self.hyperparameter_dict['num_gnn_layer'])]
        #? self.dense_final = Dense(self.hyperparameter_dict['GRU_size_prot'], activation='tanh', use_bias=True)
        self.dense_final = Dense(self.hyperparameter_dict['GRU_size_prot'], activation=None, use_bias=True)
        self.drop_out = Dropout(self.hyperparameter_dict['drop_out_rate'], input_shape=(self.hyperparameter_dict['GRU_size_prot'],))

        self.W_list = [ self.add_weight(name='W'+str(l), shape=(self.hyperparameter_dict['GRU_size_prot'], self.hyperparameter_dict['GRU_size_prot']), initializer='random_uniform', trainable=True) for l in range(self.hyperparameter_dict['num_gnn_layer'])]

    def call(self, inputs):  # Defines the computation from inputs to outputs
        amino_embd = inputs[0]
        prot_contacts = inputs[1]
        prot_contacts_self_loop = prot_contacts + K.eye(self.hyperparameter_dict['num_group']*self.hyperparameter_dict['group_size'])

        x = Reshape(target_shape=(self.hyperparameter_dict['num_group']*self.hyperparameter_dict['group_size'], self.hyperparameter_dict['GRU_size_prot']))(amino_embd)
        prot_contacts_sum = tf.einsum('bij->bi', prot_contacts_self_loop)
        prot_contacts_norm = tf.einsum('bij,bi->bij', prot_contacts_self_loop, 1/prot_contacts_sum)
        adj = prot_contacts_norm

        # x = self.bn_feat(x)
        for l in range(self.hyperparameter_dict['num_gnn_layer']):
            x0 = x
            # x = tf.einsum('bij,bjd->bid', adj, x0)

            adj_non_local = tf.einsum('bid,de,bje->bij', x0, self.W_list[l], x0)
            # adj_non_local = sigmoid(adj_non_local)
            adj_non_local = K.exp(adj_non_local)
            adj_non_local += 1e-5 * K.eye(self.hyperparameter_dict['num_group']*self.hyperparameter_dict['group_size'])
            adj_non_local = tf.einsum('bid,bid->bid', adj_non_local, prot_contacts_self_loop)
            adj_non_local_sum = tf.einsum('bij->bi', adj_non_local)
            adj_non_local_norm = tf.einsum('bij,bi->bij', adj_non_local, 1/adj_non_local_sum)
            x = tf.einsum('bij,bjd->bid', adj_non_local_norm, x0)

            for dl in range(self.hyperparameter_dict['num_dense_layer']):
                x = self.dense_list[l][dl](x)


            if self.hyperparameter_dict['drop_out']:
                x = self.drop_out(x)
            if self.hyperparameter_dict['skip_connection']:
                x = x + x0


        x = self.dense_final(x)
        x = Reshape(target_shape=(self.hyperparameter_dict['num_group'], self.hyperparameter_dict['group_size'], self.hyperparameter_dict['GRU_size_prot']))(x)
        return x

###############################



### model
## RNN for protein
prot_inter = Input(shape=(prot_max_size,comp_max_size),name='prot_inter')
prot_inter_exist = Input(shape=(1,),name='prot_inter_exist')

prot_contacts = Input(shape=(prot_max_size,prot_max_size),name='prot_contacts')
prot_data = Input(shape=(num_group,group_size))
amino_embd = TimeDistributed(Embedding(input_dim = vocab_size_protein, output_dim = GRU_size_prot, input_length=group_size))(prot_data)

""" replace with gnn
amino_lstm = TimeDistributed(GRU(units=GRU_size_prot,return_sequences=True))(amino_embd)
amino_lstm = TimeDistributed(Reshape((group_size,GRU_size_prot)))(amino_lstm)
alphas = TimeDistributed(Sep_attn_alphas(output_dim=GRU_size_prot,length=group_size),name='alphas')(amino_lstm)
prot_encoder = Sep_attn_beta(output_dim=GRU_size_prot,length=num_group)([amino_lstm,alphas])
print("alphas:",alphas.shape)
print("prot_encoder:",prot_encoder.shape)
prot_lstm = GRU(units=GRU_size_prot,return_sequences=True,name='after_GRU')(prot_encoder)
prot_lstm = Reshape((num_group,GRU_size_prot))(prot_lstm)
"""
gnn = gcn_non_local(gnn_hyperparameter_dict)
amino_gcn = gnn([amino_embd, prot_contacts])


###
gru_1 = GRU(units=GRU_size_prot,return_sequences=True)
amino_lstm = TimeDistributed(gru_1)(amino_embd)
amino_lstm = TimeDistributed(Reshape((group_size,GRU_size_prot)))(amino_lstm)
amino_lstm_1 = Reshape((40, 25, 256))(amino_lstm)
gru_11 = GRU(units=GRU_size_prot,return_sequences=True)
amino_lstm_1 = TimeDistributed(gru_11)(amino_lstm_1)
amino_lstm_1 = Reshape((25, 40, 256))(amino_lstm)

# amino_gcn = keras.layers.Add()([amino_gcn, amino_lstm_1])
# amino_gcn = weighted_add(w1=ww, w2=(1-ww))([amino_gcn, amino_lstm_1])
amino_gcn = cross_attn(w1=ww, w2=(1-ww))([amino_gcn, amino_lstm_1])
###



alphas = TimeDistributed(Sep_attn_alphas(output_dim=GRU_size_prot,length=group_size),name='alphas')(amino_gcn)
prot_lstm = Sep_attn_beta(output_dim=GRU_size_prot,length=num_group)([amino_gcn,alphas])

#? prot_lstm = Dense(GRU_size_prot, activation='tanh', use_bias=True)(prot_lstm)
prot_lstm = Dense(GRU_size_prot, activation=None, use_bias=True)(prot_lstm)



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
print("joint_attn:",joint_attn.shape)

joint_attn_1 = joint_attn_new(max_size1=num_group,shape1=GRU_size_prot,
              max_size2=comp_max_size,shape2=GRU_size_drug,name='joint_attn_new')([amino_gcn, drug_embd])


Attn = joint_vectors(dim=256)([prot_lstm,drug_embd,joint_attn])

# reg_FSGL  = FSGL_constraint(mylambda_group=l2_group_binding,mylambda_l1=l1_binding,fused_matrix=fused_matrix,mylambda_fused=l1_fused_binding,prot_max_size=prot_max_size,comp_max_size=comp_max_size)([prot_contacts,alphas,joint_attn])
reg_FSGL  = FSGL_constraint_new(mylambda_group=l2_group_binding,mylambda_l1=l1_binding,fused_matrix=fused_matrix,mylambda_fused=l1_fused_binding,prot_max_size=prot_max_size,comp_max_size=comp_max_size)([prot_contacts,joint_attn_1])
# reg_inter = interaction_penalty_new(batch=batch_size,mylambda=lambda_interaction)([prot_inter,prot_inter_exist,alphas,joint_attn_1])
reg_inter = interaction_penalty_new(batch=batch_size,mylambda=lambda_interaction)([prot_inter,prot_inter_exist,joint_attn_1])


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

if lambda_interaction == 0:
  para_num= str(int(math.log10(l1_fused_binding) * -1)) + str(int(math.log10(l2_group_binding) * -1))+ str(int(math.log10(l1_binding) * -1)) + "00"
elif lambda_interaction <= 1:
  para_num = str(int(math.log10(l1_fused_binding) * -1)) + str(int(math.log10(l2_group_binding) * -1))+ str(int(math.log10(l1_binding) * -1)) + str(int(math.log10(lambda_interaction) * -1))
else:
  para_num = str(int(math.log10(l1_fused_binding) * -1)) + str(int(math.log10(l2_group_binding) * -1))+ str(int(math.log10(l1_binding) * -1)) + str(int(lambda_interaction))
para_num += '_lr3_coldstart_newcorrectnor' + dataset + "_newembed"

model.compile(loss=penalized_loss(reg_FSGL,reg_inter),
              optimizer=optimizer)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
def scheduler(epoch):
    if epoch < 600:
        return 0.001
    elif epoch < 800:
        return 0.0001
    else:
        return 0.00001
# kscheduler = keras.callbacks.LearningRateScheduler(scheduler)
callbacks_list = [checkpoint , tensorboard]
 

# Training.
#model.load_weights("weights.best.hdf5_434")
model.fit([protein_train,compound_train_ver,compound_train_adj,prot_train_contacts,prot_train_inter,prot_train_inter_exist], IC50_train,
          batch_size=batch_size,
          epochs=200,
          verbose=1,
          validation_data=([protein_dev,compound_dev_ver,compound_dev_adj,prot_dev_contacts,prot_dev_inter,prot_dev_inter_exist], IC50_dev),
          callbacks=callbacks_list)
# model.load_weights(filepath)
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

print('pearson', scipy.stats.pearsonr(y_pred[:, 0], IC50_dev))
print(scipy.stats.kendalltau(y_pred[:, 0], IC50_dev))
print(scipy.stats.spearmanr(y_pred[:, 0], IC50_dev))
assert False

with open('record.txt', 'a+') as f:
    f.write(str(l1_fused_binding) + ' ' + str(l2_group_binding) + ' ' + str(l1_binding) + ' ' + str(lambda_interaction)  + ' ' + str(math.sqrt(mse)) + ' ')

print("std:",np.std(error))
# results = sm.OLS(y_pred,sm.add_constant(IC50_dev)).fit()
# print("R = {}".format(math.sqrt(results.rsquared)))
# print(results.summary())


print()
print("error on Test")

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
# results = sm.OLS(y_pred,sm.add_constant(test_label)).fit()
# print("R = {}".format(math.sqrt(results.rsquared)))
# print(results.summary())


import scipy
print('pearson', scipy.stats.pearsonr(y_pred[:,0], IC50_test))

np.savetxt("./logs/test_pred_affinity_" + para_num,np.asarray(y_pred))



print()
print("error on train")

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

print('pearson', scipy.stats.pearsonr(y_pred[:,0], protein_uniq_label))

np.savetxt("./logs/prot_uniq_pred_affinity_" + para_num,np.asarray(y_pred))


print()
print("error on compound unique")
compound_uniq_protein = np.load(data_processed_dir+'compound_uniq_protein.npy')
compound_uniq_compound_ver = np.load(data_processed_dir+'compound_uniq_compound_ver.npy')
compound_uniq_compound_adj = np.load(data_processed_dir+'compound_uniq_compound_adj.npy')
compound_uniq_prot_contacts = np.load(data_processed_dir+'compound_uniq_prot_contacts.npy')
compound_uniq_prot_inter = np.load(data_processed_dir+'compound_uniq_prot_inter.npy')
compound_uniq_prot_inter_exist = np.load(data_processed_dir+'compound_uniq_prot_inter_exist.npy')
compound_uniq_label = np.load(data_processed_dir+'compound_uniq_label.npy')

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

np.savetxt("./logs/double_uniq_pred_affinity_" + para_num,np.asarray(y_pred))



######### Interpretability for dev
print()
print("dev")
pred_dir = "pred_inter"
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

"""
f = open(data_dir+"/train_inter", "r")
names = []
for n in f:
    n = n.strip()
    names.append(n)
f.close()
"""

layer_name = ['alphas','joint_attn_new'] #list of layers
mylayers_output =  [model.get_layer(l).output for l in layer_name]
inp = model.input
inp.append(K.learning_phase())
functor = K.function(inp, mylayers_output )   # evaluation function

prot_dev_inter_exist = np.reshape(prot_dev_inter_exist,(len(prot_dev_inter_exist),1))
#test_comp_adj = np.reshape(test_comp_adj,(len(test_comp_adj),comp_max_size,comp_max_size))
layer_outs = functor([protein_dev,compound_dev_ver,compound_dev_adj, prot_dev_contacts, prot_dev_inter,prot_dev_inter_exist,1.])  # predict for given inputs
AP_1 = []
AUC_1 = []
AP_1_margin = []
AUC_1_margin = []


prot_dev_length = np.load('./data_processed/prot_dev_length.npy')
comp_dev_length = np.load('./data_processed/comp_dev_length.npy')

"""
count=0
for i in range(len(prot_dev_inter_exist)):
    if prot_dev_inter_exist[i] != 0:
        count += 1
        length_prot = prot_dev_length[i][0]
        length_comp = comp_dev_length[i][0]
        true_label = np.asarray(prot_dev_inter[i])[:length_prot,:length_comp]
        # true_label = np.reshape(true_label_cut,(length_prot*length_comp))

        # pred_label_1 = convert_joint_attn_matrix(np.asarray(layer_outs[0][i]),np.asarray(layer_outs[1][i]),num_group)[:length_prot,:length_comp]
        pred_label_1 = np.asarray(layer_outs[1][i])[:length_prot,:length_comp]
        # pred_label_1pred_label_1 = np.reshape(full_matrix,(length_prot*length_comp))
        # pdb.set_trace() ###

        if i % 100 == 0:
            np.save('attn_viz/label_'+str(i)+'_new.npy', true_label)
            np.save('attn_viz/pred_'+str(i)+'_new.npy', pred_label_1)

# assert False
"""


count=0
for i in range(len(prot_dev_inter_exist)):
    if prot_dev_inter_exist[i] != 0:
        count += 1
        length_prot = prot_dev_length[i][0]
        length_comp = comp_dev_length[i][0]
        true_label_cut = np.asarray(prot_dev_inter[i])[:length_prot,:length_comp]
        true_label = np.reshape(true_label_cut,(length_prot*length_comp))

        full_matrix = np.asarray(layer_outs[1][i])[:length_prot,:length_comp]
        # pdb.set_trace() ###
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


print("interaction")
mean = np.mean(AUC_1)
std = np.std(AUC_1)
print("AUC = "+str(mean) + " +/- "+str(std))
mean = np.mean(AP_1)
std = np.std(AP_1)
print("AP = "+str(mean) + " +/- "+str(std))

with open('record.txt', 'a+') as f:
    f.write(str(mean) + '\n')


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




# assert False
######### Interpretability for Test
print()
print("Test")
pred_dir = "pred_inter"
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

"""
f = open(data_dir+"/test_inter", "r")
names = []
for n in f:
    n = n.strip()
    names.append(n)

f.close()
"""

layer_name = ['alphas','joint_attn_new'] #list of layers
mylayers_output =  [model.get_layer(l).output for l in layer_name]
inp = model.input
inp.append(K.learning_phase())
functor = K.function(inp, mylayers_output )   # evaluation function

prot_test_inter_exist = np.reshape(prot_test_inter_exist,(len(prot_test_inter_exist),1))
#test_comp_adj = np.reshape(test_comp_adj,(len(test_comp_adj),comp_max_size,comp_max_size))
# layer_outs = functor([protein_test,compound_test_ver,compound_test_adj,prot_test_contacts,prot_test_contacts_true, prot_test_inter,prot_test_inter_exist, 1.])  # predict for given inputs


lll, _, _ = protein_test.shape
num_bins = math.ceil(lll/batch_size)
for i in range(num_bins):
  if i == 0:
    layer_outs = functor([protein_test[:batch_size],compound_test_ver[:batch_size],compound_test_adj[:batch_size],prot_test_contacts[:batch_size],prot_test_inter[:batch_size],prot_test_inter_exist[:batch_size], 1.])
  elif i < num_bins-1:
    temp = functor([protein_test[(i*size):((i+1)*size)],compound_test_ver[(i*size):((i+1)*size)],compound_test_adj[(i*size):((i+1)*size)],prot_test_contacts[(i*size):((i+1)*size)], prot_test_inter[(i*size):((i+1)*size)],prot_test_inter_exist[(i*size):((i+1)*size)], 1.])
    layer_outs[0] = np.concatenate((layer_outs[0],temp[0]), axis=0)
    layer_outs[1] = np.concatenate((layer_outs[1],temp[1]), axis=0)

  else:
    temp = functor([protein_test[(i*size):],compound_test_ver[(i*size):],compound_test_adj[(i*size):],prot_test_contacts[(i*size):], prot_test_inter[(i*size):],prot_test_inter_exist[(i*size):], 1.])
    layer_outs[0] = np.concatenate((layer_outs[0],temp[0]), axis=0)
    layer_outs[1] = np.concatenate((layer_outs[1],temp[1]), axis=0)




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

        # full_matrix = convert_joint_attn_matrix(np.asarray(layer_outs[0][i]),np.asarray(layer_outs[1][i]),num_group)[:length_prot,:length_comp]
        # pred_label_1 = np.reshape(full_matrix,(length_prot*length_comp))
        # cluster = cal_cluster(names[i], interaction_dir, pdb_dir, full_matrix, get_seq(protein_test[i]))
        # cluster_result += cluster

        full_matrix = np.asarray(layer_outs[1][i])[:length_prot,:length_comp]
        # pdb.set_trace() ###
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


# print("[4,8,12,16,20,+inf]:\n",cluster_result/count)
"""
with open ("./logs/test_AUPRC",'w+') as f:
    temp_str = ''
    for temp_s in AP_1_name:
        temp_str += str(temp_s) + ' '
    f.write(temp_str + '\n')
    temp_str = ''
    for temp_s in AP_1:
        temp_str += str(temp_s) + ' '
    f.write(temp_str)
"""

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


layer_name = ['alphas','joint_attn_new'] #list of layers
mylayers_output =  [model.get_layer(l).output for l in layer_name]
inp = model.input
inp.append(K.learning_phase())
functor = K.function(inp, mylayers_output )   # evaluation function


protein_uniq_prot_inter_exist = np.reshape(protein_uniq_prot_inter_exist,(len(protein_uniq_prot_inter_exist),1))
#protein_uniq_comp_adj = np.reshape(protein_uniq_comp_adj,(len(protein_uniq_comp_adj),comp_max_size,comp_max_size))
# layer_outs = functor([protein_uniq_protein,protein_uniq_compound_ver,protein_uniq_compound_adj,protein_uniq_prot_contacts,protein_uniq_prot_contacts_true,protein_uniq_prot_inter,protein_uniq_prot_inter_exist, 1.])  # predict for given inputs


lll, _, _ = protein_test.shape
num_bins = math.ceil(lll/batch_size)
for i in range(num_bins):
  if i == 0:
    layer_outs = functor([protein_uniq_protein[:batch_size],protein_uniq_compound_ver[:batch_size],protein_uniq_compound_adj[:batch_size],protein_uniq_prot_contacts[:batch_size],protein_uniq_prot_inter[:batch_size],protein_uniq_prot_inter_exist[:batch_size], 1.])
  elif i < num_bins-1:
    temp = functor([protein_uniq_protein[(i*size):((i+1)*size)],protein_uniq_compound_ver[(i*size):((i+1)*size)],protein_uniq_compound_adj[(i*size):((i+1)*size)],protein_uniq_prot_contacts[(i*size):((i+1)*size)], protein_uniq_prot_inter[(i*size):((i+1)*size)],protein_uniq_prot_inter_exist[(i*size):((i+1)*size)], 1.])
    layer_outs[0] = np.concatenate((layer_outs[0],temp[0]), axis=0)
    layer_outs[1] = np.concatenate((layer_outs[1],temp[1]), axis=0)

  else:
    temp = functor([protein_uniq_protein[(i*size):],protein_uniq_compound_ver[(i*size):],protein_uniq_compound_adj[(i*size):],protein_uniq_prot_contacts[(i*size):], protein_uniq_prot_inter[(i*size):],protein_uniq_prot_inter_exist[(i*size):], 1.])
    layer_outs[0] = np.concatenate((layer_outs[0],temp[0]), axis=0)
    layer_outs[1] = np.concatenate((layer_outs[1],temp[1]), axis=0)




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
np.savetxt("./logs/AUC_deepaffinity_protein_uniq_" + para_num + ".txt",AUC_1,delimiter=',')
np.savetxt("./logs/AP_deepaffinity_protein_uniq_" + para_num + ".txt",AP_1,delimiter=',')


######### Interpretability for compound unique
print()
print("compound unique")
pred_dir = "pred_inter"
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

layer_name = ['alphas','joint_attn_new'] #list of layers
mylayers_output =  [model.get_layer(l).output for l in layer_name]
inp = model.input
inp.append(K.learning_phase())
functor = K.function(inp, mylayers_output )   # evaluation function

compound_uniq_prot_inter_exist = np.reshape(compound_uniq_prot_inter_exist,(len(compound_uniq_prot_inter_exist),1))
#compound_uniq_comp_adj = np.reshape(compound_uniq_comp_adj,(len(compound_uniq_comp_adj),comp_max_size,comp_max_size))
layer_outs = functor([compound_uniq_protein,compound_uniq_compound_ver,compound_uniq_compound_adj, compound_uniq_prot_contacts, compound_uniq_prot_inter,compound_uniq_prot_inter_exist,1.])  # predict for given inputs
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


layer_name = ['alphas','joint_attn_new'] #list of layers
mylayers_output =  [model.get_layer(l).output for l in layer_name]
inp = model.input
inp.append(K.learning_phase())
functor = K.function(inp, mylayers_output )   # evaluation function

double_uniq_prot_inter_exist = np.reshape(double_uniq_prot_inter_exist,(len(double_uniq_prot_inter_exist),1))
#double_uniq_comp_adj = np.reshape(double_uniq_comp_adj,(len(double_uniq_comp_adj),comp_max_size,comp_max_size))
layer_outs = functor([double_uniq_protein,double_uniq_compound_ver,double_uniq_compound_adj, double_uniq_prot_contacts, double_uniq_prot_inter,double_uniq_prot_inter_exist, 1.])  # predict for given inputs
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


assert False


######### Interpretability for train
print()
print("train")
pred_dir = "pred_inter"
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)


layer_name = ['alphas','joint_attn_new'] #list of layers
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
    temp = functor([protein_train[(i*size):((i+1)*size)],compound_train_ver[(i*size):((i+1)*size)],compound_train_adj[(i*size):((i+1)*size)],prot_train_contacts[(i*size):((i+1)*size)], prot_train_inter[(i*size):((i+1)*size)],prot_train_inter_exist[(i*size):((i+1)*size)], 1.])
    layer_outs[0] = np.concatenate((layer_outs[0],temp[0]), axis=0)
    layer_outs[1] = np.concatenate((layer_outs[1],temp[1]), axis=0)

  else:
    temp = functor([protein_train[(i*size):],compound_train_ver[(i*size):],compound_train_adj[(i*size):],prot_train_contacts[(i*size):], prot_train_inter[(i*size):],prot_train_inter_exist[(i*size):], 1.])
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



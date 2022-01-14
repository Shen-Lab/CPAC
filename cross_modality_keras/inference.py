from __future__ import division, print_function, absolute_import
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
import matplotlib.pyplot as plt

import warnings

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

from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve,average_precision_score
from functools import cmp_to_key

import pdb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='concat')
parser.add_argument('--l1_fused_binding', type=float, default=0.001)
parser.add_argument('--l2_group_binding', type=float, default=0.0001)
parser.add_argument('--l1_binding', type=float, default=0.0001)
parser.add_argument('--lambda_interaction', type=float, default=10000)
parser.add_argument('--data_processed_dir', type=str, default='../data/')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--grad_clip', type=int, default=1)
parser.add_argument('--inference', type=int, default=1)
parser.add_argument('--cm_true', type=int, default=0)
args = parser.parse_args()
print(args)


l1_fused_binding = args.l1_fused_binding
l2_group_binding = args.l2_group_binding
l1_binding =  args.l1_binding
lambda_interaction = args.lambda_interaction
seed = 0
num_gnn_layer = 7

# filepath="./weights/weights/weights_" + args.model + '_' + str(l1_fused_binding) + '_' + str(l2_group_binding) + '_' + str(l1_binding) + '_' + str(int(lambda_interaction))
if args.cm_true == 1:
    filepath = './training/weights/deepmodality_' + args.model + '_true.h5'
else:
    filepath = './training/weights/deepmodality_' + args.model + '.h5'
    # filepath = './training/weights/deepmodality_' + args.model + '_gcn.h5'
    # filepath = './training/weights/deepmodality_' + args.model + '_bce.h5'
    # filepath = './training/weights/deepmodality_' + args.model + '_bce_balance.h5'


# filepath = './training/weights/deepmodality_seq_highLambda.h5'


from utils import *
from nets import *


np.random.seed(seed)
random.seed(seed)
tf.set_random_seed(seed)

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)



# load data
# train data
protein_train, compound_train_ver, compound_train_adj, prot_train_contacts, prot_train_contacts_true, prot_train_inter, prot_train_inter_exist, IC50_train = load_train_data(args.data_processed_dir)
# val data
protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts, prot_dev_contacts_true, prot_dev_inter, prot_dev_inter_exist, IC50_dev = load_val_data(args.data_processed_dir)
# test data
protein_test, compound_test_ver, compound_test_adj, prot_test_contacts, prot_test_contacts_true, prot_test_inter, prot_test_inter_exist, IC50_test = load_test_data(args.data_processed_dir)
# unseen-protein data
protein_uniq_protein, protein_uniq_compound_ver, protein_uniq_compound_adj, protein_uniq_prot_contacts, protein_uniq_prot_contacts_true, protein_uniq_prot_inter, protein_uniq_prot_inter_exist, protein_uniq_label = load_uniqProtein_data(args.data_processed_dir)
# unseen-cpmpound data
compound_uniq_protein, compound_uniq_compound_ver, compound_uniq_compound_adj, compound_uniq_prot_contacts, compound_uniq_prot_contacts_true, compound_uniq_prot_inter, compound_uniq_prot_inter_exist, compound_uniq_label = load_uniqCompound_data(args.data_processed_dir)
# unseen both data
double_uniq_protein, double_uniq_compound_ver, double_uniq_compound_adj, double_uniq_prot_contacts, double_uniq_prot_contacts_true, double_uniq_prot_inter, double_uniq_prot_inter_exist, double_uniq_label = load_uniqDouble_data(args.data_processed_dir)




fused_matrix = tf.convert_to_tensor(np.load(args.data_processed_dir+'fused_matrix.npy'))



gnn_hyperparameter_dict = {'num_gnn_layer':num_gnn_layer,
                           'num_dense_layer':2,
                           'drop_out':False,
                           'drop_out_rate':0.5,
                           'skip_connection':True,
                           'num_group':25, 'group_size':40, 'GRU_size_prot':256}

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

        for l in range(self.hyperparameter_dict['num_gnn_layer']):
            x0 = x

            '''
            prot_contacts_sum = tf.einsum('bij->bi', prot_contacts_self_loop)
            prot_contacts_norm = tf.einsum('bij,bi->bij', prot_contacts_self_loop, 1/prot_contacts_sum)
            adj = prot_contacts_norm
            x = tf.einsum('bij,bjd->bid', adj, x0)
            '''

            adj_non_local = tf.einsum('bid,de,bje->bij', x0, self.W_list[l], x0)
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



# model
prot_inter = Input(shape=(prot_max_size,comp_max_size),name='prot_inter')
prot_inter_exist = Input(shape=(1,),name='prot_inter_exist')
prot_contacts = Input(shape=(prot_max_size,prot_max_size),name='prot_contacts')
prot_data = Input(shape=(num_group,group_size))
amino_embd = TimeDistributed(Embedding(input_dim = vocab_size_protein, output_dim = GRU_size_prot, input_length=group_size))(prot_data)
if args.model == 'concat':
    # encode 2D
    gnn = gcn_non_local(gnn_hyperparameter_dict)
    amino_gcn = gnn([amino_embd, prot_contacts])
    # encode 1D
    gru_1 = GRU(units=GRU_size_prot,return_sequences=True)
    amino_lstm = TimeDistributed(gru_1)(amino_embd)
    amino_lstm = TimeDistributed(Reshape((group_size,GRU_size_prot)))(amino_lstm)
    amino_lstm_1 = Reshape((40, 25, 256))(amino_lstm)
    gru_11 = GRU(units=GRU_size_prot,return_sequences=True)
    amino_lstm_1 = TimeDistributed(gru_11)(amino_lstm_1)
    amino_lstm_1 = Reshape((25, 40, 256))(amino_lstm)
    # concat
    amino_gcn = weighted_add()([amino_gcn, amino_lstm_1])

    alphas = TimeDistributed(Sep_attn_alphas(output_dim=GRU_size_prot,length=group_size),name='alphas')(amino_gcn)
    prot_lstm = Sep_attn_beta(output_dim=GRU_size_prot,length=num_group)([amino_gcn,alphas])
    prot_lstm = Dense(GRU_size_prot, activation=None, use_bias=True)(prot_lstm)
elif args.model == 'seq':
    gru_1 = GRU(units=GRU_size_prot,return_sequences=True)
    amino_lstm = TimeDistributed(gru_1)(amino_embd)
    amino_lstm = TimeDistributed(Reshape((group_size,GRU_size_prot)))(amino_lstm)
    amino_lstm_1 = Reshape((40, 25, 256))(amino_lstm)
    gru_11 = GRU(units=GRU_size_prot,return_sequences=True)
    amino_gcn = TimeDistributed(gru_11)(amino_lstm_1)
    alphas = TimeDistributed(Sep_attn_alphas(output_dim=GRU_size_prot,length=group_size),name='alphas')(amino_lstm)
    prot_encoder = Sep_attn_beta(output_dim=GRU_size_prot,length=num_group)([amino_lstm,alphas])
    gru_2 = GRU(units=GRU_size_prot,return_sequences=True,name='after_GRU')
    prot_lstm = gru_2(prot_encoder)
    prot_lstm = Reshape((num_group,GRU_size_prot))(prot_lstm)
elif args.model == 'graph':
    gnn = gcn_non_local(gnn_hyperparameter_dict)
    amino_gcn = gnn([amino_embd, prot_contacts])
    alphas = TimeDistributed(Sep_attn_alphas(output_dim=GRU_size_prot,length=group_size),name='alphas')(amino_gcn)
    prot_lstm = Sep_attn_beta(output_dim=GRU_size_prot,length=num_group)([amino_gcn,alphas])
    prot_lstm = Dense(GRU_size_prot, activation=None, use_bias=True)(prot_lstm)
elif args.model == 'crossInteraction':
    gnn = gcn_non_local(gnn_hyperparameter_dict)
    amino_gcn = gnn([amino_embd, prot_contacts])

    gru_1 = GRU(units=GRU_size_prot,return_sequences=True)
    amino_lstm = TimeDistributed(gru_1)(amino_embd)
    amino_lstm = TimeDistributed(Reshape((group_size,GRU_size_prot)))(amino_lstm)
    amino_lstm_1 = Reshape((40, 25, 256))(amino_lstm)
    gru_11 = GRU(units=GRU_size_prot,return_sequences=True)
    amino_lstm_1 = TimeDistributed(gru_11)(amino_lstm_1)
    amino_lstm_1 = Reshape((25, 40, 256))(amino_lstm)

    amino_gcn = cross_attn()([amino_gcn, amino_lstm_1])
    alphas = TimeDistributed(Sep_attn_alphas(output_dim=GRU_size_prot,length=group_size),name='alphas')(amino_gcn)
    prot_lstm = Sep_attn_beta(output_dim=GRU_size_prot,length=num_group)([amino_gcn,alphas])
    prot_lstm = Dense(GRU_size_prot, activation=None, use_bias=True)(prot_lstm)

# encode compound
drug_data_ver = Input(shape=(comp_max_size,feature_num))
drug_data_adj = Input(shape=(comp_max_size, comp_max_size))
drug_embd = drug_data_ver
drug_embd = graph_layer(output_dim=GRU_size_drug,length=comp_max_size,input_dim=feature_num)([drug_data_adj,drug_embd])
drug_embd = graph_layer(output_dim=GRU_size_drug,length=comp_max_size,input_dim=GRU_size_drug)([drug_data_adj,drug_embd])
drug_embd = graph_layer(output_dim=GRU_size_drug,length=comp_max_size,input_dim=GRU_size_drug)([drug_data_adj,drug_embd])
drug_embd = Reshape((comp_max_size,GRU_size_drug))(drug_embd)
# interaction
joint_attn = joint_attn(max_size1=num_group,shape1=GRU_size_prot,
              max_size2=comp_max_size,shape2=GRU_size_drug,name='joint_attn')([prot_lstm,drug_embd])
joint_attn_1 = joint_attn_new(max_size1=num_group,shape1=GRU_size_prot,
              max_size2=comp_max_size,shape2=GRU_size_drug,name='joint_attn_new')([amino_gcn, drug_embd])
Attn = joint_vectors(dim=256)([prot_lstm,drug_embd,joint_attn])
# reguratization
reg_FSGL  = FSGL_constraint_new(mylambda_group=l2_group_binding,mylambda_l1=l1_binding,fused_matrix=fused_matrix,mylambda_fused=l1_fused_binding,prot_max_size=prot_max_size,comp_max_size=comp_max_size)([prot_contacts,joint_attn_1])
reg_inter = interaction_penalty_new(batch=batch_size,mylambda=lambda_interaction)([prot_inter,prot_inter_exist,joint_attn_1])
# affinity
conv_1 = Conv1D(filters=64,kernel_size=4,strides=2,padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(0.001))(Attn)
conv_1 = LeakyReLU(alpha=0.1)(conv_1)
pool_1 = MaxPooling1D(pool_size=4)(conv_1)
final = Reshape((64*32,))(pool_1)
fc_1 = Dense(units=600,kernel_initializer='glorot_uniform')(final)
fc_1 = add_fake(num=600)([fc_1,reg_FSGL])
fc_1 = add_fake(num=600)([fc_1,reg_inter])
fc_1 = LeakyReLU(alpha=0.1)(fc_1)
drop_2 = Dropout(rate=0.8)(fc_1)
fc_2 = Dense(units=300,kernel_initializer='glorot_uniform')(drop_2)
fc_2 = LeakyReLU(alpha=0.1)(fc_2)
drop_3 = Dropout(rate=0.8)(fc_2)
linear = Dense(units=1,activation="linear",kernel_initializer='glorot_uniform')(drop_3)
# pdb.set_trace()

model = Model(inputs=[prot_data,drug_data_ver,drug_data_adj,prot_contacts,prot_inter,prot_inter_exist],outputs=[linear])
if args.grad_clip == 1:
    optimizer = Adam(0.001, clipvalue=5)
else:
    optimizer = Adam(0.001)

model.compile(loss=penalized_loss(reg_FSGL,reg_inter),
              optimizer=optimizer)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=4)
tensorboard = TensorBoard(log_dir='./logs/Graph', histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [checkpoint , tensorboard]
 

if args.inference == 0:
    # Training.
    model.fit([protein_train,compound_train_ver,compound_train_adj,prot_train_contacts,prot_train_inter,prot_train_inter_exist], IC50_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              verbose=1,
              validation_data=([protein_dev,compound_dev_ver,compound_dev_adj,prot_dev_contacts,prot_dev_inter,prot_dev_inter_exist], IC50_dev),
              callbacks=callbacks_list)
else:
    model.load_weights(filepath)


if args.cm_true == 0:
    print('affinity in train')
    inputs = [protein_train, compound_train_ver, compound_train_adj, prot_train_contacts, prot_train_inter, prot_train_inter_exist]
    cal_affinity(inputs, IC50_train, model)

    print('affinity in val')
    inputs = [protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts, prot_dev_inter, prot_dev_inter_exist]
    cal_affinity(inputs, IC50_dev, model)

    print('affinity in test')
    inputs = [protein_test, compound_test_ver, compound_test_adj, prot_test_contacts, prot_test_inter, prot_test_inter_exist]
    cal_affinity(inputs, IC50_test, model)

    print('affinity in unseen proteins')
    inputs = [protein_uniq_protein, protein_uniq_compound_ver, protein_uniq_compound_adj, protein_uniq_prot_contacts, protein_uniq_prot_inter, protein_uniq_prot_inter_exist]
    cal_affinity(inputs, protein_uniq_label, model)

    print('affinity in unseen compounds')
    inputs = [compound_uniq_protein, compound_uniq_compound_ver, compound_uniq_compound_adj, compound_uniq_prot_contacts, compound_uniq_prot_inter, compound_uniq_prot_inter_exist]
    cal_affinity(inputs, compound_uniq_label, model)

    print('affinity in unseen both')
    inputs = [double_uniq_protein, double_uniq_compound_ver, double_uniq_compound_adj, double_uniq_prot_contacts, double_uniq_prot_inter, double_uniq_prot_inter_exist]
    cal_affinity(inputs, double_uniq_label, model)


    # interaction
    inp = model.input
    inp.append(keras.backend.learning_phase())
    layer_output = [model.get_layer('joint_attn_new').output]
    functor = keras.backend.function(inp, layer_output)

    print('interaction in train')
    prot_train_length = np.load(args.data_processed_dir+'prot_train_length.npy')
    comp_train_length = np.load(args.data_processed_dir+'comp_train_length.npy')
    inputs = [protein_train, compound_train_ver, compound_train_adj, prot_train_contacts, prot_train_inter, prot_train_inter_exist]
    cal_interaction(inputs, prot_train_inter, functor, args.batch_size, prot_train_length, comp_train_length)

    print('interaction in val')
    prot_dev_length = np.load(args.data_processed_dir+'prot_dev_length.npy')
    comp_dev_length = np.load(args.data_processed_dir+'comp_dev_length.npy')
    inputs = [protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts, prot_dev_inter, prot_dev_inter_exist]
    cal_interaction(inputs, prot_dev_inter, functor, args.batch_size, prot_dev_length, comp_dev_length)

    print('interaction in test')
    prot_test_length = np.load(args.data_processed_dir+'prot_test_length.npy')
    comp_test_length = np.load(args.data_processed_dir+'comp_test_length.npy')
    inputs = [protein_test, compound_test_ver, compound_test_adj, prot_test_contacts, prot_test_inter, prot_test_inter_exist.reshape((-1, 1))]
    cal_interaction(inputs, prot_test_inter, functor, args.batch_size, prot_test_length, comp_test_length)

    print('interaction in unseen proteins')
    protein_uniq_prot_length = np.load(args.data_processed_dir+'protein_uniq_prot_length.npy')
    protein_uniq_comp_length = np.load(args.data_processed_dir+'protein_uniq_comp_length.npy')
    inputs = [protein_uniq_protein, protein_uniq_compound_ver, protein_uniq_compound_adj, protein_uniq_prot_contacts, protein_uniq_prot_inter, protein_uniq_prot_inter_exist.reshape((-1, 1))]
    cal_interaction(inputs, protein_uniq_prot_inter, functor, args.batch_size, protein_uniq_prot_length, protein_uniq_comp_length)

    print('interaction in unseen compounds')
    compound_uniq_prot_length = np.load(args.data_processed_dir+'compound_uniq_prot_length.npy')
    compound_uniq_comp_length = np.load(args.data_processed_dir+'compound_uniq_comp_length.npy')
    inputs = [compound_uniq_protein, compound_uniq_compound_ver, compound_uniq_compound_adj, compound_uniq_prot_contacts, compound_uniq_prot_inter, compound_uniq_prot_inter_exist.reshape((-1, 1))]
    cal_interaction(inputs, compound_uniq_prot_inter, functor, args.batch_size, compound_uniq_prot_length, compound_uniq_comp_length)

    print('interaction in unseen both')
    double_uniq_prot_length = np.load(args.data_processed_dir+'double_uniq_prot_length.npy')
    double_uniq_comp_length = np.load(args.data_processed_dir+'double_uniq_comp_length.npy')
    inputs = [double_uniq_protein, double_uniq_compound_ver, double_uniq_compound_adj, double_uniq_prot_contacts, double_uniq_prot_inter, double_uniq_prot_inter_exist.reshape((-1, 1))]
    cal_interaction(inputs, double_uniq_prot_inter, functor, args.batch_size, double_uniq_prot_length, double_uniq_comp_length)

else:
    print('affinity in train')
    inputs = [protein_train, compound_train_ver, compound_train_adj, prot_train_contacts_true, prot_train_inter, prot_train_inter_exist]
    cal_affinity(inputs, IC50_train, model)

    print('affinity in val')
    inputs = [protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts_true, prot_dev_inter, prot_dev_inter_exist]
    cal_affinity(inputs, IC50_dev, model)

    print('affinity in test')
    inputs = [protein_test, compound_test_ver, compound_test_adj, prot_test_contacts_true, prot_test_inter, prot_test_inter_exist]
    cal_affinity(inputs, IC50_test, model)

    print('affinity in unseen proteins')
    inputs = [protein_uniq_protein, protein_uniq_compound_ver, protein_uniq_compound_adj, protein_uniq_prot_contacts_true, protein_uniq_prot_inter, protein_uniq_prot_inter_exist]
    cal_affinity(inputs, protein_uniq_label, model)

    print('affinity in unseen compounds')
    inputs = [compound_uniq_protein, compound_uniq_compound_ver, compound_uniq_compound_adj, compound_uniq_prot_contacts_true, compound_uniq_prot_inter, compound_uniq_prot_inter_exist]
    cal_affinity(inputs, compound_uniq_label, model)

    print('affinity in unseen both')
    inputs = [double_uniq_protein, double_uniq_compound_ver, double_uniq_compound_adj, double_uniq_prot_contacts_true, double_uniq_prot_inter, double_uniq_prot_inter_exist]
    cal_affinity(inputs, double_uniq_label, model)


    # interaction
    inp = model.input
    inp.append(keras.backend.learning_phase())
    layer_output = [model.get_layer('joint_attn_new').output]
    functor = keras.backend.function(inp, layer_output)

    print('interaction in train')
    prot_train_length = np.load(args.data_processed_dir+'prot_train_length.npy')
    comp_train_length = np.load(args.data_processed_dir+'comp_train_length.npy')
    inputs = [protein_train, compound_train_ver, compound_train_adj, prot_train_contacts_true, prot_train_inter, prot_train_inter_exist]
    cal_interaction(inputs, prot_train_inter, functor, args.batch_size, prot_train_length, comp_train_length)

    print('interaction in val')
    prot_dev_length = np.load(args.data_processed_dir+'prot_dev_length.npy')
    comp_dev_length = np.load(args.data_processed_dir+'comp_dev_length.npy')
    inputs = [protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts_true, prot_dev_inter, prot_dev_inter_exist]
    cal_interaction(inputs, prot_dev_inter, functor, args.batch_size, prot_dev_length, comp_dev_length)

    print('interaction in test')
    prot_test_length = np.load(args.data_processed_dir+'prot_test_length.npy')
    comp_test_length = np.load(args.data_processed_dir+'comp_test_length.npy')
    inputs = [protein_test, compound_test_ver, compound_test_adj, prot_test_contacts_true, prot_test_inter, prot_test_inter_exist.reshape((-1, 1))]
    cal_interaction(inputs, prot_test_inter, functor, args.batch_size, prot_test_length, comp_test_length)

    print('interaction in unseen proteins')
    protein_uniq_prot_length = np.load(args.data_processed_dir+'protein_uniq_prot_length.npy')
    protein_uniq_comp_length = np.load(args.data_processed_dir+'protein_uniq_comp_length.npy')
    inputs = [protein_uniq_protein, protein_uniq_compound_ver, protein_uniq_compound_adj, protein_uniq_prot_contacts_true, protein_uniq_prot_inter, protein_uniq_prot_inter_exist.reshape((-1, 1))]
    cal_interaction(inputs, protein_uniq_prot_inter, functor, args.batch_size, protein_uniq_prot_length, protein_uniq_comp_length)

    print('interaction in unseen compounds')
    compound_uniq_prot_length = np.load(args.data_processed_dir+'compound_uniq_prot_length.npy')
    compound_uniq_comp_length = np.load(args.data_processed_dir+'compound_uniq_comp_length.npy')
    inputs = [compound_uniq_protein, compound_uniq_compound_ver, compound_uniq_compound_adj, compound_uniq_prot_contacts_true, compound_uniq_prot_inter, compound_uniq_prot_inter_exist.reshape((-1, 1))]
    cal_interaction(inputs, compound_uniq_prot_inter, functor, args.batch_size, compound_uniq_prot_length, compound_uniq_comp_length)

    print('interaction in unseen both')
    double_uniq_prot_length = np.load(args.data_processed_dir+'double_uniq_prot_length.npy')
    double_uniq_comp_length = np.load(args.data_processed_dir+'double_uniq_comp_length.npy')
    inputs = [double_uniq_protein, double_uniq_compound_ver, double_uniq_compound_adj, double_uniq_prot_contacts_true, double_uniq_prot_inter, double_uniq_prot_inter_exist.reshape((-1, 1))]
    cal_interaction(inputs, double_uniq_prot_inter, functor, args.batch_size, double_uniq_prot_length, double_uniq_comp_length)


protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts, IC50_dev = load_JAK2()
print('affinity in JAK2')
inputs = [protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts, prot_dev_inter, prot_dev_inter_exist]
cal_affinity(inputs, IC50_dev, model)

protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts, IC50_dev = load_TIE2()
print('affinity in TIE2')
inputs = [protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts, prot_dev_inter, prot_dev_inter_exist]
cal_affinity(inputs, IC50_dev, model)


print('identity and tanimoto')
seq_identity = np.load(args.data_processed_dir+'unseenboth_identity.npy')
comp_tanimoto = np.load(args.data_processed_dir+'unseenboth_tanimoto.npy')

print('identity')
inputs = [double_uniq_protein, double_uniq_compound_ver, double_uniq_compound_adj, double_uniq_prot_contacts, double_uniq_prot_inter, double_uniq_prot_inter_exist]
input_seq1 = [i[seq_identity<=0.3] for i in inputs]
input_seq2 = [i[(seq_identity>0.3) & (seq_identity<=0.6)] for i in inputs]
input_seq3 = [i[seq_identity>0.6] for i in inputs]
cal_affinity(input_seq1, double_uniq_label[seq_identity<=0.3], model)
cal_affinity(input_seq2, double_uniq_label[(seq_identity>0.3) & (seq_identity<=0.6)], model)
cal_affinity(input_seq3, double_uniq_label[seq_identity>0.6], model)

inputs = [double_uniq_protein, double_uniq_compound_ver, double_uniq_compound_adj, double_uniq_prot_contacts, double_uniq_prot_inter, double_uniq_prot_inter_exist.reshape((-1, 1))]
input_seq1 = [i[seq_identity<=0.3] for i in inputs]
input_seq2 = [i[(seq_identity>0.3) & (seq_identity<=0.6)] for i in inputs]
input_seq3 = [i[seq_identity>0.6] for i in inputs]
cal_interaction(input_seq1, double_uniq_prot_inter[seq_identity<=0.3], functor, args.batch_size, double_uniq_prot_length[seq_identity<=0.3], double_uniq_comp_length[seq_identity<=0.3])
cal_interaction(input_seq2, double_uniq_prot_inter[(seq_identity>0.3) & (seq_identity<=0.6)], functor, args.batch_size, double_uniq_prot_length[(seq_identity>0.3) & (seq_identity<=0.6)], double_uniq_comp_length[(seq_identity>0.3) & (seq_identity<=0.6)])
cal_interaction(input_seq3, double_uniq_prot_inter[seq_identity>0.6], functor, args.batch_size, double_uniq_prot_length[seq_identity>0.6], double_uniq_comp_length[seq_identity>0.6])

print('tanimoto')
inputs = [double_uniq_protein, double_uniq_compound_ver, double_uniq_compound_adj, double_uniq_prot_contacts, double_uniq_prot_inter, double_uniq_prot_inter_exist]
input_seq1 = [i[comp_tanimoto<=0.5] for i in inputs]
input_seq2 = [i[(comp_tanimoto>0.5) & (comp_tanimoto<=0.8)] for i in inputs]
input_seq3 = [i[comp_tanimoto>0.8] for i in inputs]
cal_affinity(input_seq1, double_uniq_label[comp_tanimoto<=0.5], model)
cal_affinity(input_seq2, double_uniq_label[(comp_tanimoto>0.5) & (comp_tanimoto<=0.8)], model)
cal_affinity(input_seq3, double_uniq_label[comp_tanimoto>0.8], model)

inputs = [double_uniq_protein, double_uniq_compound_ver, double_uniq_compound_adj, double_uniq_prot_contacts, double_uniq_prot_inter, double_uniq_prot_inter_exist.reshape((-1, 1))]
input_seq1 = [i[comp_tanimoto<=0.5] for i in inputs]
input_seq2 = [i[(comp_tanimoto>0.5) & (comp_tanimoto<=0.8)] for i in inputs]
input_seq3 = [i[comp_tanimoto>0.8] for i in inputs]
cal_interaction(input_seq1, double_uniq_prot_inter[comp_tanimoto<=0.5], functor, args.batch_size, double_uniq_prot_length[comp_tanimoto<=0.5], double_uniq_comp_length[comp_tanimoto<=0.5])
cal_interaction(input_seq2, double_uniq_prot_inter[(comp_tanimoto>0.5) & (comp_tanimoto<=0.8)], functor, args.batch_size, double_uniq_prot_length[(comp_tanimoto>0.5) & (comp_tanimoto<=0.8)], double_uniq_comp_length[(comp_tanimoto>0.5) & (comp_tanimoto<=0.8)])
cal_interaction(input_seq3, double_uniq_prot_inter[comp_tanimoto>0.8], functor, args.batch_size, double_uniq_prot_length[comp_tanimoto>0.8], double_uniq_comp_length[comp_tanimoto>0.8])


# case study
inp = model.input
inp.append(keras.backend.learning_phase())
layer_output = [model.get_layer('joint_attn_new').output]
functor = keras.backend.function(inp, layer_output)

protein_uniq_prot_length = np.load(args.data_processed_dir+'protein_uniq_prot_length.npy')
protein_uniq_comp_length = np.load(args.data_processed_dir+'protein_uniq_comp_length.npy')
compound_uniq_prot_length = np.load(args.data_processed_dir+'compound_uniq_prot_length.npy')
compound_uniq_comp_length = np.load(args.data_processed_dir+'compound_uniq_comp_length.npy')

print('CA2 and AL1')
n = 580
inputs = [protein_uniq_protein[[n]], protein_uniq_compound_ver[[n]], protein_uniq_compound_adj[[n]], protein_uniq_prot_contacts[[n]], protein_uniq_prot_inter[[n]], protein_uniq_prot_inter_exist[[n]]]
cal_affinity(inputs, protein_uniq_label[[n]], model)
inputs = [protein_uniq_protein[[n]], protein_uniq_compound_ver[[n]], protein_uniq_compound_adj[[n]], protein_uniq_prot_contacts[[n]], protein_uniq_prot_inter[[n]], protein_uniq_prot_inter_exist.reshape((-1, 1))[[n]]]
cal_interaction(inputs, protein_uniq_prot_inter[[n]], functor, args.batch_size, protein_uniq_prot_length[[n]], protein_uniq_comp_length[[n]])
print('CA2 and TI2')
n = 592
inputs = [protein_uniq_protein[[n]], protein_uniq_compound_ver[[n]], protein_uniq_compound_adj[[n]], protein_uniq_prot_contacts[[n]], protein_uniq_prot_inter[[n]], protein_uniq_prot_inter_exist[[n]]]
cal_affinity(inputs, protein_uniq_label[[n]], model)
inputs = [protein_uniq_protein[[n]], protein_uniq_compound_ver[[n]], protein_uniq_compound_adj[[n]], protein_uniq_prot_contacts[[n]], protein_uniq_prot_inter[[n]], protein_uniq_prot_inter_exist.reshape((-1, 1))[[n]]]
cal_interaction(inputs, protein_uniq_prot_inter[[n]], functor, args.batch_size, protein_uniq_prot_length[[n]], protein_uniq_comp_length[[n]])
print('PYGM and CPB')
n = 5
inputs = [compound_uniq_protein[[n]], compound_uniq_compound_ver[[n]], compound_uniq_compound_adj[[n]], compound_uniq_prot_contacts[[n]], compound_uniq_prot_inter[[n]], compound_uniq_prot_inter_exist[[n]]]
cal_affinity(inputs, compound_uniq_label[[n]], model)
inputs = [compound_uniq_protein[[n]], compound_uniq_compound_ver[[n]], compound_uniq_compound_adj[[n]], compound_uniq_prot_contacts[[n]], compound_uniq_prot_inter[[n]], compound_uniq_prot_inter_exist.reshape((-1, 1))[[n]]]
cal_interaction(inputs, compound_uniq_prot_inter[[n]], functor, args.batch_size, compound_uniq_prot_length[[n]], compound_uniq_comp_length[[n]])
print('PYGM and T68')
n = 6
inputs = [compound_uniq_protein[[n]], compound_uniq_compound_ver[[n]], compound_uniq_compound_adj[[n]], compound_uniq_prot_contacts[[n]], compound_uniq_prot_inter[[n]], compound_uniq_prot_inter_exist[[n]]]
cal_affinity(inputs, compound_uniq_label[[n]], model)
inputs = [compound_uniq_protein[[n]], compound_uniq_compound_ver[[n]], compound_uniq_compound_adj[[n]], compound_uniq_prot_contacts[[n]], compound_uniq_prot_inter[[n]], compound_uniq_prot_inter_exist.reshape((-1, 1))[[n]]]
cal_interaction(inputs, compound_uniq_prot_inter[[n]], functor, args.batch_size, compound_uniq_prot_length[[n]], compound_uniq_comp_length[[n]])
print('LCK and LHL')
n = 791
inputs = [protein_uniq_protein[[n]], protein_uniq_compound_ver[[n]], protein_uniq_compound_adj[[n]], protein_uniq_prot_contacts[[n]], protein_uniq_prot_inter[[n]], protein_uniq_prot_inter_exist[[n]]]
cal_affinity(inputs, protein_uniq_label[[n]], model)
inputs = [protein_uniq_protein[[n]], protein_uniq_compound_ver[[n]], protein_uniq_compound_adj[[n]], protein_uniq_prot_contacts[[n]], protein_uniq_prot_inter[[n]], protein_uniq_prot_inter_exist.reshape((-1, 1))[[n]]]
cal_interaction(inputs, protein_uniq_prot_inter[[n]], functor, args.batch_size, protein_uniq_prot_length[[n]], protein_uniq_comp_length[[n]])








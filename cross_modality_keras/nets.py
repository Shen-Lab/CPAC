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

from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem, ChemicalFeatures, rdchem
from rdkit import RDConfig
import rdkit.Chem.rdPartialCharges as rdPartialCharges

from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve,average_precision_score
from functools import cmp_to_key

import pdb


class joint_attn(Layer):

    def __init__(self, max_size1,shape1,max_size2,shape2, **kwargs):
        self.max_size1 = max_size1
        self.max_size2 = max_size2
        self.shape1 = shape1
        self.shape2 = shape2
        super(joint_attn, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='W',
                                      shape=(self.shape1,self.shape2),
                                      initializer='random_uniform',
                                      trainable=True)
        self.b = self.add_weight(name='b',
                                      shape=(self.max_size1,self.max_size2),
                                      initializer='random_uniform',
                                      trainable=True)

        super(joint_attn, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        V = tf.einsum('bij,jk->bik',x,self.W)
        joint_attn = tf.nn.softmax(tf.tanh(tf.einsum('bkj,bij->bik',y,V)+self.b))
        return joint_attn

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.max_size1,self.max_size2)



class joint_attn_new(Layer):

    def __init__(self, max_size1,shape1,max_size2,shape2, **kwargs):
        self.max_size1 = max_size1
        self.max_size2 = max_size2
        self.shape1 = shape1
        self.shape2 = shape2
        super(joint_attn_new, self).__init__(**kwargs)

    def build(self, input_shape):
        self.pd1 = Dense(256, activation='relu', use_bias=True)
        self.pd2 = Dense(256, activation=None, use_bias=True)
        self.cd1 = Dense(256, activation='relu', use_bias=True)
        super(joint_attn_new, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        x = Reshape(target_shape=(1000,256))(x)

        x = self.pd1(x)
        x = self.pd2(x)
        y = self.cd1(y)

        joint_attn = tf.exp(tf.einsum('bkj,bij->bik', y, x))
        ja_sum = tf.einsum('bij->b', joint_attn)
        joint_attn = tf.einsum('bij,b->bij', joint_attn, 1/ja_sum)
        return joint_attn

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.max_size1,self.max_size2)



class joint_vectors(Layer):

    def __init__(self, dim, **kwargs):
        self.dim = dim
        super(joint_vectors, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                      shape=(self.dim,input_shape[0][2]),
                                      initializer='random_uniform',
                                      trainable=True)
        self.W2 = self.add_weight(name='W2',
                                      shape=(self.dim,input_shape[1][2]),
                                      initializer='random_uniform',
                                      trainable=True)

        self.b = self.add_weight(name='b',
                                      shape=(self.dim,),
                                      initializer='random_uniform',
                                      trainable=True)

        super(joint_vectors, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        joint_attn = inputs[2]
        prot = tf.einsum('bij,kj->bik',x,self.W1)
        drug = tf.einsum('bij,kj->bik',y,self.W2)
        vec = tf.tanh(tf.einsum('bik,bjk->bijk',prot,drug)+self.b)
        Attn = tf.expand_dims(tf.einsum('bijk,bij->bk',vec,joint_attn),2)
        return Attn

    def compute_output_shape(self, input_shape):
        return  (input_shape[0][0],self.dim,1)



class Sep_attn_alphas(Layer):

    def __init__(self, output_dim,length, **kwargs):
        self.output_dim = output_dim
        self.length = length
        super(Sep_attn_alphas, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='W', 
                                      shape=(input_shape[2],self.output_dim),
                                      initializer='random_uniform',
                                      trainable=True)
        self.b = self.add_weight(name='b',          
                                      shape=(input_shape[2],),
                                      initializer='random_uniform',
                                      trainable=True)
        self.U = self.add_weight(name='U',          
                                      shape=(self.output_dim,1),
                                      initializer='random_uniform',
                                      trainable=True)
        super(Sep_attn_alphas, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        V = tf.tanh(tf.tensordot(x,self.W,axes=1)+self.b)
        V = Reshape((self.length,self.output_dim))(V)
        VU = tf.squeeze(tf.tensordot(V,self.U,axes=[[2],[0]]),axis=2) 
        VU = Reshape((self.length,))(VU)
        alphas = tf.nn.softmax(VU,name='alphas')
        #alphas = tf.expand_dims(alphas,2)
        #Attn = tf.expand_dims(tf.reduce_sum(x *tf.expand_dims(alphas,-1),1),2)
        return alphas

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.length)


class Sep_attn_beta(Layer):

    def __init__(self, output_dim,length, **kwargs):
        self.output_dim = output_dim
        self.length = length
        super(Sep_attn_beta, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Sep_attn_beta, self).build(input_shape)  # Be sure to call this at the end

    def call(self, y, **kwargs):
        x = y[0]
        alphas = y[1]
        #Attn = tf.expand_dims(tf.reduce_sum(x *tf.expand_dims(alphas,-1),1),2)
        Attn = tf.einsum('bijk,bij->bik',x,alphas)
        return Attn

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],self.length,self.output_dim)


class Sep_attn(Layer):

    def __init__(self, output_dim,length, **kwargs):
        self.output_dim = output_dim
        self.length = length
        super(Sep_attn, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='W',
                                      shape=(input_shape[2],self.output_dim),
                                      initializer='random_uniform',
                                      trainable=True)
        self.b = self.add_weight(name='b',
                                      shape=(input_shape[2],),
                                      initializer='random_uniform',
                                      trainable=True)
        self.U = self.add_weight(name='U',
                                      shape=(self.output_dim,1),
                                      initializer='random_uniform',
                                      trainable=True)

        super(Sep_attn, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        V = tf.tanh(tf.tensordot(x,self.W,axes=1)+self.b)
        V = Reshape((self.length,self.output_dim))(V)
        VU = tf.squeeze(tf.tensordot(V,self.U,axes=[[2],[0]]),axis=2)
        VU = Reshape((self.length,))(VU)
        alphas = tf.nn.softmax(VU,name='alphas')
        Attn = tf.expand_dims(tf.reduce_sum(x *tf.expand_dims(alphas,-1),1),2)
        return Attn

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class graph_layer(Layer):

    def __init__(self,output_dim,length,input_dim, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.length = length
        super(graph_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                      shape=(self.input_dim,self.output_dim),
                                      initializer='random_uniform',
                                      trainable=True)

        super(graph_layer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        A = inputs[0]
        X = inputs[1]
        Z = tf.einsum('bij,bjk->bik', A, X)
        Y = tf.nn.relu(tf.einsum('bij,jk->bik', Z, self.W))
        return Y

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.length,self.output_dim)


class FSGL_constraint(Layer):

    def __init__(self,mylambda_group,mylambda_l1, fused_matrix,mylambda_fused, prot_max_size,comp_max_size,**kwargs):
        self.mylambda_group = mylambda_group
        self.mylambda_l1 = mylambda_l1
        self.fused_matrix = fused_matrix
        self.mylambda_fused = mylambda_fused
        self.prot_max_size = prot_max_size
        self.comp_max_size = comp_max_size
        super(FSGL_constraint, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FSGL_constraint, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        real = inputs[0]
        alpha = inputs[1]
        group_att = inputs[2]

        binding = Reshape((prot_max_size,comp_max_size))(tf.einsum('bij,bik->bijk',alpha,group_att))
        reg_l1 = K.sum(K.sum(K.abs(binding),axis=1),axis=1) # l1 penalty with whole matrix
        #reg_fused  = K.sum(K.sum(tf.abs(tf.add(binding,-tf.manip.roll(binding,shift=1,axis=2))),axis=2),axis=1) # fussed penalty for newer version of tf (>1.6)
        reg_group  = K.sum(multiply([tf.sqrt(K.sum(real,axis=2)),tf.sqrt(K.sum(tf.nn.relu(tf.einsum('bij,bki->bjk',tf.square(binding),real)),axis=1))]),axis=1) # group penalty
        reg_fused = K.sum(K.sum(K.abs(tf.einsum('bij,ti->bjt',binding,self.fused_matrix)),axis=2),axis=1)


        return self.mylambda_l1 * reg_l1 + self.mylambda_group * reg_group + self.mylambda_fused * reg_fused


    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)

class interaction_penalty(Layer):

    def __init__(self,batch,mylambda,**kwargs):
        self.mylambda = mylambda
        self.batch = batch
        super(interaction_penalty, self).__init__(**kwargs)

    def build(self, input_shape):
        super(interaction_penalty, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        real = inputs[0]
        real_exist = inputs[1]
        real_exist = tf.squeeze(real_exist,axis=-1)
        alpha = inputs[2]
        group_att = inputs[3]
        binding = Reshape((prot_max_size,comp_max_size))(tf.einsum('bij,bik->bijk',alpha,group_att))
        #eps = 0.0001
        #const = self.batch/(K.sum(real_exist,axis=0)+eps)
        #penalty = multiply([real_exist,tf.sqrt(K.sum(K.sum(tf.square(add([real,-binding])),axis=2),axis=1))])

        real = tf.einsum('b,bij->bij',real_exist,real)
        penalty = tf.sqrt(K.sum(K.sum(tf.square(add([real,-binding])),axis=2),axis=1))

        # pdb.set_trace()

        return self.mylambda*penalty

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)


class FSGL_constraint_new(Layer):
    def __init__(self,mylambda_group,mylambda_l1, fused_matrix,mylambda_fused, prot_max_size,comp_max_size,**kwargs):
        self.mylambda_group = mylambda_group
        self.mylambda_l1 = mylambda_l1
        self.fused_matrix = fused_matrix
        self.mylambda_fused = mylambda_fused
        self.prot_max_size = prot_max_size
        self.comp_max_size = comp_max_size
        super(FSGL_constraint_new, self).__init__(**kwargs)
    def build(self, input_shape):
        super(FSGL_constraint_new, self).build(input_shape)  # Be sure to call this at the end
    def call(self, inputs):
        real = inputs[0]
        binding = inputs[1]
        # binding = Reshape((prot_max_size,comp_max_size))(tf.einsum('bij,bik->bijk',alpha,group_att))
        reg_l1 = K.sum(K.sum(K.abs(binding),axis=1),axis=1) # l1 penalty with whole matrix
        #reg_fused  = K.sum(K.sum(tf.abs(tf.add(binding,-tf.manip.roll(binding,shift=1,axis=2))),axis=2),axis=1) # fussed penalty for newer version of tf (>1.6)
        reg_group  = K.sum(multiply([tf.sqrt(K.sum(real,axis=2)), tf.sqrt(K.sum(tf.nn.relu(tf.einsum('bij,bki->bjk',tf.square(binding),real)),axis=1))]),axis=1) # group penalty
        # pdb.set_trace()
        reg_fused = K.sum(K.sum(K.abs(tf.einsum('bij,ti->bjt',binding,self.fused_matrix)),axis=2),axis=1)
        return self.mylambda_l1 * reg_l1 + self.mylambda_group * reg_group + self.mylambda_fused * reg_fused
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)



class interaction_penalty_new(Layer):

    def __init__(self,batch,mylambda,**kwargs):
        self.mylambda = mylambda
        self.batch = batch
        super(interaction_penalty_new, self).__init__(**kwargs)

    def build(self, input_shape):
        super(interaction_penalty_new, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        real = inputs[0]
        real_exist = inputs[1]
        real_exist = tf.squeeze(real_exist,axis=-1)
        binding = inputs[2]
        # binding = Reshape((prot_max_size,comp_max_size))(tf.einsum('bij,bik->bijk',alpha,group_att))
        #eps = 0.0001
        #const = self.batch/(K.sum(real_exist,axis=0)+eps)
        #penalty = multiply([real_exist,tf.sqrt(K.sum(K.sum(tf.square(add([real,-binding])),axis=2),axis=1))])

        real = tf.einsum('b,bij->bij',real_exist,real)
        penalty = tf.sqrt(K.sum(K.sum(tf.square(add([real,-binding])),axis=2),axis=1))

        return self.mylambda*penalty

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)




class add_fake(Layer):

    def __init__(self,num,**kwargs):
        self.num = num
        super(add_fake, self).__init__(**kwargs)

    def build(self, input_shape):
        super(add_fake, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        output = inputs[0]+0*tf.tile(tf.expand_dims(inputs[1],axis=-1),[1,self.num])
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],self.num)


class weighted_add(Layer):

    def __init__(self, **kwargs):
        super(weighted_add, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1', 
                                      shape=(256, 256),
                                      initializer='random_uniform',
                                      trainable=True)
        self.W2 = self.add_weight(name='W2', 
                                      shape=(256, 256),
                                      initializer='random_uniform',
                                      trainable=True)
        super(weighted_add, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        output = tf.einsum('baij,jk->baik', inputs[0], self.W1) + tf.einsum('baij,jk->baik', inputs[1], self.W2)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class cross_attn(Layer):

    def __init__(self, **kwargs):
        super(cross_attn, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1', 
                                      shape=(256, 256),
                                      initializer='random_uniform',
                                      trainable=True)
        self.W2 = self.add_weight(name='W2', 
                                      shape=(256, 256),
                                      initializer='random_uniform',
                                      trainable=True)

        self.attn1 = self.add_weight(name='A1', 
                                      shape=(256, 256),
                                      initializer='random_uniform',
                                      trainable=True)
        self.attn2 = self.add_weight(name='A2', 
                                      shape=(256, 256),
                                      initializer='random_uniform',
                                      trainable=True)


        super(cross_attn, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        # pdb.set_trace()
        i0, i1 = inputs[0], inputs[1]

        inter0 = sigmoid(tf.einsum('baij,jk,baik->bai', i1, self.attn1, i0))
        inter1 = sigmoid(tf.einsum('baij,jk,baik->bai', i0, self.attn2, i1))
        i0 += tf.einsum('baij,bai->baij', i0, inter0)
        i1 += tf.einsum('baij,bai->baij', i1, inter1)
        output = tf.einsum('baij,jk->baik', i0, self.W1) + tf.einsum('baij,jk->baik', i1, self.W2)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]



def penalized_loss(reg_fsgl,reg_inter):
    def loss(y_true, y_pred):
        mse = K.square(y_pred - y_true)
        #inter_penalty = K.in_train_phase(K.mean(reg_inter,axis=-1),tf.constant(0,dtype=tf.float32))
        inter_penalty = K.mean(reg_inter,axis=-1)
        return K.mean(mse + reg_fsgl, axis=-1) + inter_penalty
    return loss


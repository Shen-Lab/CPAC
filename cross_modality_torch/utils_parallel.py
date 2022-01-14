import numpy as np
import scipy
import math
import pdb

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
## dictionary compound
index_aa = {0:'_PAD',1:'_GO',2:'_EOS',3:'_UNK',4:'A',5:'R',6:'N',7:'D',8:'C',9:'Q',10:'E',11:'G',12:'H',13:'I',14:'L',15:'K',16:'M',17:'F',18:'P',19:'S',20:'T',21:'W',22:'Y',23:'V'}
pdb2single = {"GLY":"G","ALA":"A","SER":"S","THR":"T","CYS":"C","VAL":"V","LEU":"L","ILE":"I","MET":"M","PRO":"P",\
"PHE":"F","TYR":"Y","TRP":"W","ASP":"D","GLU":"E","ASN":"N","GLN":"Q","HIS":"H","LYS":"K","ARG":"R", "UNK":"X",\
"SEC":"U","PYL":"O","MSE":"M","CAS":"C","SGB":"S","CGA":"E","TRQ":"W","TPO":"T","SEP":"S","CME":"C","FT6":"W","OCS":"C","SUN":"S","SXE":"S"}
dict_atom = {'C':0,'N':1,'O':2,'S':3,'F':4,'Si':5,'Cl':6,'P':7,'Br':8,'I':9,'B':10,'Unknown':11,'_PAD':12}
dict_atom_hal_don = {'C':0,'N':0,'O':0,'S':0,'F':1,'Si':0,'Cl':2,'P':0,'Br':3,'I':4,'B':0,'Unknown':0,'_PAD':0}
# dict_atom_hybridization = {Chem.rdchem.HybridizationType.SP:0,Chem.rdchem.HybridizationType.SP2:1,Chem.rdchem.HybridizationType.SP3:2,Chem.rdchem.HybridizationType.SP3D:3,Chem.rdchem.HybridizationType.SP3D2:4}

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
# _WORD_SPLIT = re.compile(b"(\S)")
# _WORD_SPLIT_2 = re.compile(b",")
# _DIGIT_RE = re.compile(br"\d")
group_size = 40
num_group = 25
# feature_num = len(dict_atom) + 24+ len(dict_atom_hybridization) + 1
feature_num = 43 
full_prot_size = group_size * num_group
prot_max_size = group_size * num_group
comp_max_size = 56


def load_train_data(data_processed_dir):
    protein_train = np.load(data_processed_dir+'protein_train.npy')
    compound_train_ver = np.load(data_processed_dir+'compound_train_ver.npy')
    compound_train_adj = np.load(data_processed_dir+'compound_train_adj.npy')
    prot_train_contacts = np.load(data_processed_dir+'prot_train_contacts.npy')
    prot_train_contacts_true = np.load(data_processed_dir+'prot_train_contacts_true.npy')
    prot_train_inter = np.load(data_processed_dir+'prot_train_inter.npy')
    prot_train_inter_exist = np.load(data_processed_dir+'prot_train_inter_exist.npy')
    IC50_train = np.load(data_processed_dir+'IC50_train.npy')
    return protein_train, compound_train_ver, compound_train_adj, prot_train_contacts, prot_train_contacts_true, prot_train_inter, prot_train_inter_exist, IC50_train


def load_val_data(data_processed_dir):
    protein_dev = np.load(data_processed_dir+'protein_dev.npy')
    compound_dev_ver = np.load(data_processed_dir+'compound_dev_ver.npy')
    compound_dev_adj = np.load(data_processed_dir+'compound_dev_adj.npy')
    prot_dev_contacts = np.load(data_processed_dir+'prot_dev_contacts.npy')
    prot_dev_contacts_true = np.load(data_processed_dir+'prot_dev_contacts_true.npy')
    prot_dev_inter = np.load(data_processed_dir+'prot_dev_inter.npy')
    prot_dev_inter_exist = np.load(data_processed_dir+'prot_dev_inter_exist.npy')
    IC50_dev = np.load(data_processed_dir+'IC50_dev.npy')
    return protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts, prot_dev_contacts_true, prot_dev_inter, prot_dev_inter_exist, IC50_dev


def load_test_data(data_processed_dir):
    protein_test = np.load(data_processed_dir+'protein_test.npy')
    compound_test_ver = np.load(data_processed_dir+'compound_test_ver.npy')
    compound_test_adj = np.load(data_processed_dir+'compound_test_adj.npy')
    prot_test_contacts = np.load(data_processed_dir+'prot_test_contacts.npy')
    prot_test_contacts_true = np.load(data_processed_dir+'prot_test_contacts_true.npy')
    prot_test_inter = np.load(data_processed_dir+'prot_test_inter.npy')
    prot_test_inter_exist = np.load(data_processed_dir+'prot_test_inter_exist.npy')
    IC50_test = np.load(data_processed_dir+'IC50_test.npy')
    return protein_test, compound_test_ver, compound_test_adj, prot_test_contacts, prot_test_contacts_true, prot_test_inter, prot_test_inter_exist, IC50_test


def load_uniqProtein_data(data_processed_dir):
    protein_uniq_protein = np.load(data_processed_dir+'protein_uniq_protein.npy')
    protein_uniq_compound_ver = np.load(data_processed_dir+'protein_uniq_compound_ver.npy')
    protein_uniq_compound_adj = np.load(data_processed_dir+'protein_uniq_compound_adj.npy')
    protein_uniq_prot_contacts = np.load(data_processed_dir+'protein_uniq_prot_contacts.npy')
    protein_uniq_prot_contacts_true = np.load(data_processed_dir+'protein_uniq_prot_contacts_true.npy')
    protein_uniq_prot_inter = np.load(data_processed_dir+'protein_uniq_prot_inter.npy')
    protein_uniq_prot_inter_exist = np.load(data_processed_dir+'protein_uniq_prot_inter_exist.npy')
    protein_uniq_label = np.load(data_processed_dir+'protein_uniq_label.npy')
    return protein_uniq_protein, protein_uniq_compound_ver, protein_uniq_compound_adj, protein_uniq_prot_contacts, protein_uniq_prot_contacts_true, protein_uniq_prot_inter, protein_uniq_prot_inter_exist, protein_uniq_label


def load_uniqCompound_data(data_processed_dir):
    compound_uniq_protein = np.load(data_processed_dir+'compound_uniq_protein.npy')
    compound_uniq_compound_ver = np.load(data_processed_dir+'compound_uniq_compound_ver.npy')
    compound_uniq_compound_adj = np.load(data_processed_dir+'compound_uniq_compound_adj.npy')
    compound_uniq_prot_contacts = np.load(data_processed_dir+'compound_uniq_prot_contacts.npy')
    compound_uniq_prot_contacts_true = np.load(data_processed_dir+'compound_uniq_prot_contacts_true.npy')
    compound_uniq_prot_inter = np.load(data_processed_dir+'compound_uniq_prot_inter.npy')
    compound_uniq_prot_inter_exist = np.load(data_processed_dir+'compound_uniq_prot_inter_exist.npy')
    compound_uniq_label = np.load(data_processed_dir+'compound_uniq_label.npy')
    return compound_uniq_protein, compound_uniq_compound_ver, compound_uniq_compound_adj, compound_uniq_prot_contacts, compound_uniq_prot_contacts_true, compound_uniq_prot_inter, compound_uniq_prot_inter_exist, compound_uniq_label


def load_uniqDouble_data(data_processed_dir):
    double_uniq_protein = np.load(data_processed_dir+'double_uniq_protein.npy')
    double_uniq_compound_ver = np.load(data_processed_dir+'double_uniq_compound_ver.npy')
    double_uniq_compound_adj = np.load(data_processed_dir+'double_uniq_compound_adj.npy')
    double_uniq_prot_contacts = np.load(data_processed_dir+'double_uniq_prot_contacts.npy')
    double_uniq_prot_contacts_true = np.load(data_processed_dir+'double_uniq_prot_contacts_true.npy')
    double_uniq_prot_inter = np.load(data_processed_dir+'double_uniq_prot_inter.npy')
    double_uniq_prot_inter_exist = np.load(data_processed_dir+'double_uniq_prot_inter_exist.npy')
    double_uniq_label = np.load(data_processed_dir+'double_uniq_label.npy')
    return double_uniq_protein, double_uniq_compound_ver, double_uniq_compound_adj, double_uniq_prot_contacts, double_uniq_prot_contacts_true, double_uniq_prot_inter, double_uniq_prot_inter_exist, double_uniq_label


def load_JAK2():
    dp = '/scratch/user/yuning.you/proj/DeepRelations/HierRNN_GCN_constraint_sup/data_processed/sar/JAK2/'
    protein_dev = np.load(dp+'protein.npy')
    prot_dev_contacts = np.load(dp+'protein_adj.npy')
    compound_dev_ver = np.load(dp+'compound.npy')
    compound_dev_adj = np.load(dp+'compound_adj.npy')
    IC50_dev = np.load(dp+'labels.npy')
    return protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts, IC50_dev


def load_TIE2():
    dp = '/scratch/user/yuning.you/proj/DeepRelations/HierRNN_GCN_constraint_sup/data_processed/sar/TIE2/'
    protein_dev = np.load(dp+'protein.npy')
    prot_dev_contacts = np.load(dp+'protein_adj.npy')
    compound_dev_ver = np.load(dp+'compound.npy')
    compound_dev_adj = np.load(dp+'compound_adj.npy')
    IC50_dev = np.load(dp+'labels.npy')
    return protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts, IC50_dev


def cal_affinity(inputs, labels, model, logging=False, logpath=''):
    N = labels.shape[0]
    y_pred = np.asarray(model.predict(inputs))
    mse = 0
    for n in range(N):
        mse += (y_pred[n] - labels[n]) ** 2
    mse /= N
    rmse = np.sqrt(mse)[0]
    # pdb.set_trace()
    pearson, _ = scipy.stats.pearsonr(y_pred.squeeze(), labels.squeeze())
    tau, _ = scipy.stats.kendalltau(y_pred.squeeze(), labels.squeeze())
    rho, _ = scipy.stats.spearmanr(y_pred.squeeze(), labels.squeeze())
    print('rmse', rmse, 'pearson', pearson, 'tau', tau, 'rho', rho)
    if logging:
        with open(logpath, 'a+') as f:
            f.write(str(rmse) + ' ' + str(pearson) + ' ')


from sklearn.metrics import roc_curve, auc, average_precision_score
def cal_interaction(inputs, labels, functor, batch_size, prot_length, comp_length, logging=False, logpath=''):
    N = labels.shape[0]
    NN = math.ceil(N / batch_size)
    for n in range(NN):
        if n == 0:
            inputn = [i[:batch_size] for i in inputs]
            inputn.append(1.)
            outputs = functor(inputn)[0]
        elif n < NN - 1:
            inputn = [i[(n*batch_size):((n+1)*batch_size)] for i in inputs]
            inputn.append(1.)
            outputs = np.concatenate((outputs, functor(inputn)[0]), axis=0)
        else:
            inputn = [i[(n*batch_size):] for i in inputs]
            inputn.append(1.)
            outputs = np.concatenate((outputs, functor(inputn)[0]), axis=0)

    AP = []
    AUC = []
    AP_margin = []
    AUC_margin = []
    count=0
    for i in range(N):
        if inputs[-1][i] != 0:
            count += 1
            length_prot = int(prot_length[i])
            length_comp = int(comp_length[i])
            true_label_cut = np.asarray(labels[i])[:length_prot, :length_comp]
            true_label = np.reshape(true_label_cut, (length_prot*length_comp))

            full_matrix = np.asarray(outputs[i])[:length_prot, :length_comp]
            pred_label = np.reshape(full_matrix, (length_prot*length_comp))

            average_precision_whole = average_precision_score(true_label, pred_label)
            AP.append(average_precision_whole)
            fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label)
            roc_auc_whole = auc(fpr_whole, tpr_whole)
            AUC.append(roc_auc_whole)

            true_label = np.amax(true_label_cut,axis=1)
            pred_label = np.amax(full_matrix,axis=1)

            average_precision_whole = average_precision_score(true_label, pred_label)
            AP_margin.append(average_precision_whole)

            fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label)
            roc_auc_whole = auc(fpr_whole, tpr_whole)
            AUC_margin.append(roc_auc_whole)

    print('interaction auprc', np.mean(AP), 'auroc', np.mean(AUC), 'binding site auprc', np.mean(AP_margin), 'auroc', np.mean(AUC_margin))
    if logging:
        with open(logpath, 'a+') as f:
            f.write(str(np.mean(AP)) + ' ' + str(np.mean(AP_margin)) + ' ')


import torch
def cal_affinity_torch(model, loader, logging=False, logpath=''):
    y_pred, labels = np.zeros(len(loader.dataset)), np.zeros(len(loader.dataset))
    batch = 0
    for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in loader:
        prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label = prot_data.cuda(), drug_data_ver.cuda(), drug_data_adj.cuda(), prot_contacts.cuda(), prot_inter.cuda(), prot_inter_exist.cuda(), label.cuda()
        with torch.no_grad():
            _, affn = model.forward_inter_affn(prot_data, drug_data_ver, drug_data_adj, prot_contacts)

        if batch != len(loader.dataset) // 32:
            labels[batch*32:(batch+1)*32] = label.squeeze().cpu().numpy()
            y_pred[batch*32:(batch+1)*32] = affn.squeeze().detach().cpu().numpy()
        else:
            labels[batch*32:] = label.squeeze().cpu().numpy()
            y_pred[batch*32:] = affn.squeeze().detach().cpu().numpy()
        batch += 1

    N = labels.shape[0]
    # y_pred = np.asarray(model.predict(inputs))
    mse = 0
    for n in range(N):
        mse += (y_pred[n] - labels[n]) ** 2
    mse /= N
    rmse = np.sqrt(mse)
    # pdb.set_trace()
    pearson, _ = scipy.stats.pearsonr(y_pred.squeeze(), labels.squeeze())
    tau, _ = scipy.stats.kendalltau(y_pred.squeeze(), labels.squeeze())
    rho, _ = scipy.stats.spearmanr(y_pred.squeeze(), labels.squeeze())
    print('rmse', rmse, 'pearson', pearson, 'tau', tau, 'rho', rho)
    if logging:
        with open(logpath, 'a+') as f:
            f.write(str(rmse) + ' ' + str(pearson) + ' ')


def cal_interaction_torch(model, loader, prot_length, comp_length, logging=False, logpath=''):
    outputs, labels, ind = np.zeros((len(loader.dataset), 1000, 56)), np.zeros((len(loader.dataset), 1000, 56)), np.zeros(len(loader.dataset))
    batch = 0
    for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in loader:
        prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label = prot_data.cuda(), drug_data_ver.cuda(), drug_data_adj.cuda(), prot_contacts.cuda(), prot_inter.cuda(), prot_inter_exist.cuda(), label.cuda()
        with torch.no_grad():
            inter, _ = model.forward_inter_affn(prot_data, drug_data_ver, drug_data_adj, prot_contacts)

        if batch != len(loader.dataset) // 32:
            labels[batch*32:(batch+1)*32] = prot_inter.cpu().numpy()
            outputs[batch*32:(batch+1)*32] = inter.detach().cpu().numpy()
            ind[batch*32:(batch+1)*32] = prot_inter_exist.cpu().numpy()
        else:
            labels[batch*32:] = prot_inter.cpu().numpy()
            outputs[batch*32:] = inter.detach().cpu().numpy()
            ind[batch*32:] = prot_inter_exist.cpu().numpy()
        batch += 1

    batch_size = 32
    N = labels.shape[0]
    '''
    NN = math.ceil(N / batch_size)
    for n in range(NN):
        if n == 0:
            inputn = [i[:batch_size] for i in inputs]
            inputn.append(1.)
            outputs = functor(inputn)[0]
        elif n < NN - 1:
            inputn = [i[(n*batch_size):((n+1)*batch_size)] for i in inputs]
            inputn.append(1.)
            outputs = np.concatenate((outputs, functor(inputn)[0]), axis=0)
        else:
            inputn = [i[(n*batch_size):] for i in inputs]
            inputn.append(1.)
            outputs = np.concatenate((outputs, functor(inputn)[0]), axis=0)
    '''

    AP = []
    AUC = []
    AP_margin = []
    AUC_margin = []
    count=0
    for i in range(N):
        if ind[i] != 0:
            count += 1
            length_prot = int(prot_length[i])
            length_comp = int(comp_length[i])
            true_label_cut = np.asarray(labels[i])[:length_prot, :length_comp]
            true_label = np.reshape(true_label_cut, (length_prot*length_comp))

            full_matrix = np.asarray(outputs[i])[:length_prot, :length_comp]
            pred_label = np.reshape(full_matrix, (length_prot*length_comp))

            average_precision_whole = average_precision_score(true_label, pred_label)
            AP.append(average_precision_whole)
            fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label)
            roc_auc_whole = auc(fpr_whole, tpr_whole)
            AUC.append(roc_auc_whole)

            true_label = np.amax(true_label_cut,axis=1)
            pred_label = np.amax(full_matrix,axis=1)

            average_precision_whole = average_precision_score(true_label, pred_label)
            AP_margin.append(average_precision_whole)

            fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label)
            roc_auc_whole = auc(fpr_whole, tpr_whole)
            AUC_margin.append(roc_auc_whole)

    print('interaction auprc', np.mean(AP), 'auroc', np.mean(AUC), 'binding site auprc', np.mean(AP_margin), 'auroc', np.mean(AUC_margin))
    if logging:
        with open(logpath, 'a+') as f:
            f.write(str(np.mean(AP)) + ' ' + str(np.mean(AP_margin)) + ' ')




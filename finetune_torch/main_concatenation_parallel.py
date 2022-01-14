import torch
import torch.nn as nn

from utils_parallel import *
import pdb
import argparse
import random
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--l0', type=float, default=0.001)
parser.add_argument('--l1', type=float, default=0.001)
parser.add_argument('--l2', type=float, default=0.001)
parser.add_argument('--l3', type=float, default=1000.0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--train', type=int, default=0)
parser.add_argument('--data_processed_dir', type=str, default='../data/')

parser.add_argument('--ss_seq', type=str, default='mask') # none, mask, aug
parser.add_argument('--ss_graph', type=str, default='mask')
parser.add_argument('--p_seq_mask', type=float, default=0.15)
parser.add_argument('--p_graph_mask', type=float, default=0.15)
parser.add_argument('--p_seq_replace', type=float, default=0.15)
parser.add_argument('--p_seq_sub', type=float, default=0.3)
parser.add_argument('--p_graph_drop', type=float, default=0.15)
parser.add_argument('--p_graph_sub', type=float, default=0.3)

parser.add_argument('--supervised', type=int, default=0)
args = parser.parse_args()
print(args)


weight_pretrain_pth = ''
if args.ss_seq == 'none':
    weight_pretrain_pth += 'seq_none_'
elif args.ss_seq == 'mask':
    weight_pretrain_pth += ('seq_mask_' + str(args.p_seq_mask) + '_')
elif args.ss_seq == 'aug':
    weight_pretrain_pth += ('seq_aug_' + str(args.p_seq_replace) + '_' + str(args.p_seq_sub) + '_')
else:
    print('seq ss error')
    assert False

if args.ss_graph == 'none':
    weight_pretrain_pth += 'graph_none'
elif args.ss_graph == 'mask':
    weight_pretrain_pth += ('graph_mask_' + str(args.p_graph_mask))
elif args.ss_graph == 'aug':
    weight_pretrain_pth += ('graph_aug_' + str(args.p_graph_drop) + '_' + str(args.p_graph_sub))
else:
    print('seq ss error')
    assert False


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
np.random.seed(0)
random.seed(0)


###### define network ######
class net_crossInteraction(nn.Module):
    def __init__(self, lambda_l1, lambda_fused, lambda_group, lambda_bind):
        super().__init__()
        self.aminoAcid_embedding = nn.Embedding(29, 256)
        self.gru0 = nn.GRU(256, 256, batch_first=True)
        self.gru1 = nn.GRU(256, 256, batch_first=True)
        self.gat = net_prot_gat()
        self.crossInteraction = crossInteraction()
        self.gcn_comp = net_comp_gcn()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.joint_attn_prot, self.joint_attn_comp = nn.Linear(256, 256), nn.Linear(256, 256)
        self.tanh = nn.Tanh()

        self.regressor0 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
                                       nn.LeakyReLU(0.1),
                                       nn.MaxPool1d(kernel_size=4, stride=4))
        self.regressor1 = nn.Sequential(nn.Linear(64*32, 600),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.5),
                                        nn.Linear(600, 300),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.5),
                                        nn.Linear(300, 1))

        self.lambda_l1, self.lambda_fused, self.lambda_group = lambda_l1, lambda_fused, lambda_group
        self.lambda_bind = lambda_bind

    def forward(self, prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label):
        # protein embedding
        aminoAcid_embedding = self.aminoAcid_embedding(prot_data)

        b, i, j, d = aminoAcid_embedding.size()
        prot_seq_embedding = aminoAcid_embedding.reshape(b*i, j, d)
        prot_seq_embedding, _ = self.gru0(prot_seq_embedding)
        prot_seq_embedding = prot_seq_embedding.reshape(b*j, i, d)
        prot_seq_embedding, _ = self.gru1(prot_seq_embedding)
        prot_seq_embedding = prot_seq_embedding.reshape(b, i*j, d)

        prot_graph_embedding = aminoAcid_embedding.reshape(b, i*j, d)
        prot_graph_embedding = self.gat(prot_graph_embedding, prot_contacts)

        prot_embedding = self.crossInteraction(prot_seq_embedding, prot_graph_embedding)

        # compound embedding
        comp_embedding = self.gcn_comp(drug_data_ver, drug_data_adj)

        # compound-protein interaction
        inter_comp_prot = self.sigmoid(torch.einsum('bij,bkj->bik', self.joint_attn_prot(self.relu(prot_embedding)), self.joint_attn_comp(self.relu(comp_embedding))))
        inter_comp_prot_sum = torch.einsum('bij->b', inter_comp_prot)
        inter_comp_prot = torch.einsum('bij,b->bij', inter_comp_prot, 1/inter_comp_prot_sum)

        # compound-protein joint embedding
        cp_embedding = self.tanh(torch.einsum('bij,bkj->bikj', prot_embedding, comp_embedding))
        cp_embedding = torch.einsum('bijk,bij->bk', cp_embedding, inter_comp_prot)

        # compound-protein affinity
        affn_comp_prot = cp_embedding[:, None, :]
        affn_comp_prot = self.regressor0(affn_comp_prot)
        affn_comp_prot = affn_comp_prot.view(b, 64*32)
        affn_comp_prot = self.regressor1(affn_comp_prot)

        loss0, loss1, loss2 = self.loss_reg(inter_comp_prot, fused_matrix.to(inter_comp_prot.device), prot_contacts), self.loss_inter(inter_comp_prot, prot_inter, prot_inter_exist), self.loss_affn(affn_comp_prot, label)
        loss = loss0 + loss1 + loss2

        return loss

    def forward_inter_affn(self, prot_data, drug_data_ver, drug_data_adj, prot_contacts):
        # protein embedding
        aminoAcid_embedding = self.aminoAcid_embedding(prot_data)

        b, i, j, d = aminoAcid_embedding.size()
        prot_seq_embedding = aminoAcid_embedding.reshape(b*i, j, d)
        prot_seq_embedding, _ = self.gru0(prot_seq_embedding)
        prot_seq_embedding = prot_seq_embedding.reshape(b*j, i, d)
        prot_seq_embedding, _ = self.gru1(prot_seq_embedding)
        prot_seq_embedding = prot_seq_embedding.reshape(b, i*j, d)

        prot_graph_embedding = aminoAcid_embedding.reshape(b, i*j, d)
        prot_graph_embedding = self.gat(prot_graph_embedding, prot_contacts)

        prot_embedding = self.crossInteraction(prot_seq_embedding, prot_graph_embedding)

        # compound embedding
        comp_embedding = self.gcn_comp(drug_data_ver, drug_data_adj)

        # compound-protein interaction
        inter_comp_prot = self.sigmoid(torch.einsum('bij,bkj->bik', self.joint_attn_prot(self.relu(prot_embedding)), self.joint_attn_comp(self.relu(comp_embedding))))
        inter_comp_prot_sum = torch.einsum('bij->b', inter_comp_prot)
        inter_comp_prot = torch.einsum('bij,b->bij', inter_comp_prot, 1/inter_comp_prot_sum)

        # compound-protein joint embedding
        cp_embedding = self.tanh(torch.einsum('bij,bkj->bikj', prot_embedding, comp_embedding))
        cp_embedding = torch.einsum('bijk,bij->bk', cp_embedding, inter_comp_prot)

        # compound-protein affinity
        affn_comp_prot = cp_embedding[:, None, :]
        affn_comp_prot = self.regressor0(affn_comp_prot)
        affn_comp_prot = affn_comp_prot.view(b, 64*32)
        affn_comp_prot = self.regressor1(affn_comp_prot)

        return inter_comp_prot, affn_comp_prot

    def loss_reg(self, inter, fused_matrix, prot_contacts):
        reg_l1 = torch.abs(inter).sum(dim=(1,2)).mean()
        reg_fused = torch.abs(torch.einsum('bij,ti->bjt', inter, fused_matrix)).sum(dim=(1,2)).mean()
        # reg_group = ( torch.sqrt(torch.einsum('bij,bki->bjk', inter**2, prot_contacts).sum(dim=1)) * torch.sqrt(prot_contacts.sum(dim=2)) ).sum(dim=1).mean()
        group = torch.einsum('bij,bki->bjk', inter**2, prot_contacts).sum(dim=1)
        group[group==0] = group[group==0] + 1e10
        reg_group = ( torch.sqrt(group) * torch.sqrt(prot_contacts.sum(dim=2)) ).sum(dim=1).mean()
        # reg_group = ( torch.einsum('bij,bki->bjk', inter**2, prot_contacts).sum(dim=1) * prot_contacts.sum(dim=2) ).sum(dim=1).mean()

        reg_loss = self.lambda_l1 * reg_l1 + self.lambda_fused * reg_fused + self.lambda_group * reg_group
        return reg_loss

    def loss_inter(self, inter, prot_inter, prot_inter_exist):
        label = torch.einsum('b,bij->bij', prot_inter_exist, prot_inter)
        loss = torch.sqrt(((inter - label) ** 2).sum(dim=(1,2))).mean() * self.lambda_bind
        return loss

    def loss_affn(self, affn, label):
        loss = ((affn - label) ** 2).mean()
        return loss


class net_prot_gat(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.ModuleList([nn.Linear(256, 256) for _ in range(7)])
        self.linear1 = nn.ModuleList([nn.Linear(256, 256) for _ in range(7)])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.w_attn = nn.ModuleList([nn.Linear(256, 256) for _ in range(7)])
        self.linear_final = nn.Linear(256, 256)

    def forward(self, x, adj):
        adj[:, list(range(1000)), list(range(1000))] = 1
        for l in range(7):
            x0 = x

            adj_attn = self.sigmoid(torch.einsum('bij,bkj->bik', self.w_attn[l](x), x))
            adj_attn = adj_attn + 1e-5 * torch.eye(1000).to(x.device)
            adj_attn = torch.einsum('bij,bij->bij', adj_attn, adj)
            adj_attn_sum = torch.einsum('bij->bi', adj_attn)
            adj_attn = torch.einsum('bij,bi->bij', adj_attn, 1/adj_attn_sum)

            x = torch.einsum('bij,bjd->bid', adj_attn, x)
            x = self.relu(self.linear0[l](x))
            x = self.relu(self.linear1[l](x))

            x += x0

        x = self.linear_final(x)
        return x


class crossInteraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.crossInteraction0 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
        self.crossInteraction1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(512, 256)

    def forward(self, x_seq, x_graph):
        '''
        CI0 = self.tanh(torch.einsum('bij,bij->bi', self.crossInteraction0(x_graph), x_seq)) + 1
        CI1 = self.tanh(torch.einsum('bij,bij->bi', self.crossInteraction1(x_seq), x_graph)) + 1
        x_seq = torch.einsum('bij,bi->bij', x_seq, CI0)
        x_graph = torch.einsum('bij,bi->bij', x_graph, CI1)
        '''

        x = torch.cat((x_seq, x_graph), dim=2)
        x = self.linear(x)
        return x


class net_comp_gcn(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.ModuleList([nn.Linear(43, 256), nn.Linear(256, 256), nn.Linear(256, 256)])
        self.relu = nn.ReLU()
        self.linear_final = nn.Linear(256, 256)

    def forward(self, x, adj):
        for l in range(3):
            x = self.linear[l](x)
            x = torch.einsum('bij,bjd->bid', adj, x)
            x = self.relu(x)
        x = self.linear_final(x)
        return x


class dataset(torch.utils.data.Dataset):
    def __init__(self, name_split='train'):
        if name_split == 'train':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, _, self.prot_inter, self.prot_inter_exist, self.label = load_train_data(args.data_processed_dir)
        elif name_split == 'val':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, _, self.prot_inter, self.prot_inter_exist, self.label = load_val_data(args.data_processed_dir)
        elif name_split == 'test':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, _, self.prot_inter, self.prot_inter_exist, self.label = load_test_data(args.data_processed_dir)
        elif name_split == 'unseen_prot':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, _, self.prot_inter, self.prot_inter_exist, self.label = load_uniqProtein_data(args.data_processed_dir)
        elif name_split == 'unseen_comp':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, _, self.prot_inter, self.prot_inter_exist, self.label = load_uniqCompound_data(args.data_processed_dir)
        elif name_split == 'unseen_both':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, _, self.prot_inter, self.prot_inter_exist, self.label = load_uniqDouble_data(args.data_processed_dir)

        self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, self.prot_inter, self.prot_inter_exist, self.label = torch.tensor(self.prot_data), torch.tensor(self.drug_data_ver).float().float(), torch.tensor(self.drug_data_adj).float(), torch.tensor(self.prot_contacts).float(), torch.tensor(self.prot_inter).float(), torch.tensor(self.prot_inter_exist).float().squeeze().float(), torch.tensor(self.label).float()
    def __len__(self):
        return self.prot_data.size()[0]
    def __getitem__(self, index):
        return self.prot_data[index], self.drug_data_ver[index], self.drug_data_adj[index], self.prot_contacts[index], self.prot_inter[index], self.prot_inter_exist[index], self.label[index]



###### train ######
train_set = dataset('train')
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
val_set = dataset('val')
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)
model = net_crossInteraction(args.l0, args.l1, args.l2, args.l3)
if args.supervised == 0:
    model.load_state_dict(torch.load('../pretrain_torch/weights/' + weight_pretrain_pth + '.pth'))
elif args.supervised == 1:
    model.load_state_dict(torch.load('../CPAC_pretrain/weights/concatenation_supervised_' + weight_pretrain_pth + '.pth'))
model = nn.DataParallel(model)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

fused_matrix = torch.tensor(np.load(args.data_processed_dir+'fused_matrix.npy')).cuda()
loss_val_best = 1e10

if args.train == 1:
    # train
    for epoch in range(args.epoch):
        model.train()
        loss_epoch, batch = 0, 0
        for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in train_loader:
            prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label = prot_data.cuda(), drug_data_ver.cuda(), drug_data_adj.cuda(), prot_contacts.cuda(), prot_inter.cuda(), prot_inter_exist.cuda(), label.cuda()

            optimizer.zero_grad()
            loss = model(prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label).mean()
            # print('epoch', epoch, 'batch', batch, loss.detach().cpu().numpy())

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step()
            loss_epoch += loss.detach().cpu().numpy()
            batch += 1

        model.eval()
        loss_epoch_val, batch_val = 0, 0
        for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in val_loader:
            prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label = prot_data.cuda(), drug_data_ver.cuda(), drug_data_adj.cuda(), prot_contacts.cuda(), prot_inter.cuda(), prot_inter_exist.cuda(), label.cuda()
            with torch.no_grad():
                loss = model(prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label).mean()
            loss_epoch_val += loss.detach().cpu().numpy()
            batch_val += 1

        print('epoch', epoch, 'train loss', loss_epoch/batch, 'val loss', loss_epoch_val/batch_val)
        if loss_epoch_val/batch_val < loss_val_best:
            loss_val_best = loss_epoch_val/batch_val
            if args.supervised == 0:
                torch.save(model.module.state_dict(), './weights/concatenation_' +  weight_pretrain_pth + '.pth')
            elif args.supervised == 1:
                torch.save(model.module.state_dict(), './weights/concatenation_supervised_' + weight_pretrain_pth + '.pth')

del train_loader
del val_loader


###### evaluation ######
# evaluation
model = net_crossInteraction(args.l0, args.l1, args.l2, args.l3).cuda()
if args.supervised == 0:
    model.load_state_dict(torch.load('./weights/concatenation_' +  weight_pretrain_pth + '.pth'))
else:
    model.load_state_dict(torch.load('./weights/concatenation_supervised_' +  weight_pretrain_pth + '.pth'))
model.eval()

data_processed_dir = args.data_processed_dir

print('train')
eval_set = dataset('train')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length = np.load(data_processed_dir+'prot_train_length.npy')
comp_length = np.load(data_processed_dir+'comp_train_length.npy')
cal_interaction_torch(model, eval_loader, prot_length, comp_length)

print('val')
with open('./logs/hpo_concatenation.log', 'a+') as f:
    f.write(str(args.l0) + ' ' + str(args.l1) + ' ' + str(args.l2) + ' ' + str(args.l3) + ' ')
eval_set = dataset('val')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader, logging=True, logpath='./logs/hpo_concatenation.log')
prot_length = np.load(data_processed_dir+'prot_dev_length.npy')
comp_length = np.load(data_processed_dir+'comp_dev_length.npy')
cal_interaction_torch(model, eval_loader, prot_length, comp_length, logging=True, logpath='./logs/hpo_concatenation.log')
with open('./logs/hpo_concatenation.log', 'a+') as f:
    f.write('\n')

print('test')
eval_set = dataset('test')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length = np.load(data_processed_dir+'prot_test_length.npy')
comp_length = np.load(data_processed_dir+'comp_test_length.npy')
cal_interaction_torch(model, eval_loader, prot_length, comp_length)

print('unseen protein')
eval_set = dataset('unseen_prot')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length = np.load(data_processed_dir+'protein_uniq_prot_length.npy')
comp_length = np.load(data_processed_dir+'protein_uniq_comp_length.npy')
cal_interaction_torch(model, eval_loader, prot_length, comp_length)

print('unseen compound')
eval_set = dataset('unseen_comp')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length = np.load(data_processed_dir+'compound_uniq_prot_length.npy')
comp_length = np.load(data_processed_dir+'compound_uniq_comp_length.npy')
cal_interaction_torch(model, eval_loader, prot_length, comp_length)

print('unseen both')
eval_set = dataset('unseen_both')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length = np.load(data_processed_dir+'double_uniq_prot_length.npy')
comp_length = np.load(data_processed_dir+'double_uniq_comp_length.npy')
cal_interaction_torch(model, eval_loader, prot_length, comp_length)


idx_seqIdentity = np.load(data_processed_dir+'unseenboth_identity.npy')
idx_seq1 = np.arange(202)[idx_seqIdentity<=0.3]
idx_seq2 = np.arange(202)[(idx_seqIdentity>0.3) & (idx_seqIdentity<=0.6)]
idx_seq3 = np.arange(202)[idx_seqIdentity>0.6]
idx_compTanimoto = np.load(data_processed_dir+'unseenboth_tanimoto.npy')
idx_comp1 = np.arange(202)[idx_compTanimoto<=0.5]
idx_comp2 = np.arange(202)[(idx_compTanimoto>0.5) & (idx_compTanimoto<=0.8)]
idx_comp3 = np.arange(202)[idx_compTanimoto>0.8]

class dataset_simSplit(torch.utils.data.Dataset):
    def __init__(self, idx=None):
        self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, _, self.prot_inter, self.prot_inter_exist, self.label = load_uniqDouble_data(args.data_processed_dir)
        self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, self.prot_inter, self.prot_inter_exist, self.label = torch.tensor(self.prot_data)[idx], torch.tensor(self.drug_data_ver).float()[idx], torch.tensor(self.drug_data_adj).float()[idx], torch.tensor(self.prot_contacts).float()[idx], torch.tensor(self.prot_inter).float()[idx], torch.tensor(self.prot_inter_exist).float().squeeze().float()[idx], torch.tensor(self.label).float()[idx]
    def __len__(self):
        return self.prot_data.size()[0]
    def __getitem__(self, index):
        return self.prot_data[index], self.drug_data_ver[index], self.drug_data_adj[index], self.prot_contacts[index], self.prot_inter[index], self.prot_inter_exist[index], self.label[index]

print('sequence identity 1')
eval_set = dataset_simSplit(idx_seq1)
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length = np.load(data_processed_dir+'double_uniq_prot_length.npy')[idx_seq1]
comp_length = np.load(data_processed_dir+'double_uniq_comp_length.npy')[idx_seq1]
cal_interaction_torch(model, eval_loader, prot_length, comp_length)

print('sequence identity 2')
eval_set = dataset_simSplit(idx_seq2)
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length = np.load(data_processed_dir+'double_uniq_prot_length.npy')[idx_seq2]
comp_length = np.load(data_processed_dir+'double_uniq_comp_length.npy')[idx_seq2]
cal_interaction_torch(model, eval_loader, prot_length, comp_length)

print('sequence identity 3')
eval_set = dataset_simSplit(idx_seq3)
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length = np.load(data_processed_dir+'double_uniq_prot_length.npy')[idx_seq3]
comp_length = np.load(data_processed_dir+'double_uniq_comp_length.npy')[idx_seq3]
cal_interaction_torch(model, eval_loader, prot_length, comp_length)

print('compound tanimoto 1')
eval_set = dataset_simSplit(idx_comp1)
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length = np.load(data_processed_dir+'double_uniq_prot_length.npy')[idx_comp1]
comp_length = np.load(data_processed_dir+'double_uniq_comp_length.npy')[idx_comp1]
cal_interaction_torch(model, eval_loader, prot_length, comp_length)

print('compound tanimoto 2')
eval_set = dataset_simSplit(idx_comp2)
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length = np.load(data_processed_dir+'double_uniq_prot_length.npy')[idx_comp2]
comp_length = np.load(data_processed_dir+'double_uniq_comp_length.npy')[idx_comp2]
cal_interaction_torch(model, eval_loader, prot_length, comp_length)

print('compound tanimoto 3')
eval_set = dataset_simSplit(idx_comp3)
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length = np.load(data_processed_dir+'double_uniq_prot_length.npy')[idx_comp3]
comp_length = np.load(data_processed_dir+'double_uniq_comp_length.npy')[idx_comp3]
cal_interaction_torch(model, eval_loader, prot_length, comp_length)


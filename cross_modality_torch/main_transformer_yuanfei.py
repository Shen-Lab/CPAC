import torch
import torch.nn as nn

from utils_parallel import *
import pdb
import argparse
import random
from tqdm import tqdm
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--l0', type=float, default=0.0001)
parser.add_argument('--l1', type=float, default=0.001)
parser.add_argument('--l2', type=float, default=0.01)
parser.add_argument('--l3', type=float, default=1000.0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--train', type=int, default=0)
parser.add_argument('--data_processed_dir', type=str, default='/scratch/user/yuning.you/proj/DeepRelations/HierRNN_GCN_constraint_sup/data_processed/')
parser.add_argument('--transformer_config', type=str, default=1)
args = parser.parse_args()
print(args)


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

        self.prot_linear = nn.Linear(768, 256)

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
        '''
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
        '''

        prot_embedding = self.prot_linear(prot_data)
        b = prot_embedding.shape[0]

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
        '''
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
        '''

        prot_embedding = self.prot_linear(prot_data)
        b = prot_embedding.shape[0]

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
        CI0 = self.tanh(torch.einsum('bij,bij->bi', self.crossInteraction0(x_graph), x_seq)) + 1
        CI1 = self.tanh(torch.einsum('bij,bij->bi', self.crossInteraction1(x_seq), x_graph)) + 1
        x_seq = torch.einsum('bij,bi->bij', x_seq, CI0)
        x_graph = torch.einsum('bij,bi->bij', x_graph, CI1)

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


trans_conf_dict = {'1':'embedding_all_bert_1_rp15.pickle', '2':'embedding_all_bert_2_rp15.pickle', '3':'embedding_all_bert_3_rp15.pickle', '4':'embedding_all_bert_4_rp15.pickle', '5':'embedding_all_bert_1_rp75.pickle'}
with open('/scratch/user/yuning.you/dataset/cpac_transformer_embeddings/embeddings/'+trans_conf_dict[args.transformer_config], 'rb') as f:
    data = pickle.load(f)
with open('/scratch/user/yuning.you/dataset/cpac_transformer_embeddings/embeddings/all_seqs.txt', 'r') as f:
    seqData = f.read().split('\n')[:-1]
seq_emb_list = []
for n, seq in enumerate(tqdm(seqData)):
    n += 1
    # seq
    token_dict = {'_PAD':0, '_GO':1, '_EOS':2, '_UNK':3, 'A': 4, 'R': 5, 'N': 6, 'D': 7, 'C': 8, 'Q': 9, 'E': 10, 'G': 11, 'H': 12, 'I': 13, 'L': 14, 'K': 15, 'M': 16, 'F': 17, 'P': 18, 'S': 19, 'T': 20, 'W': 21, 'Y': 22, 'V': 23, 'X': 24, 'U': 25, 'O': 26, 'B': 27, 'Z': 28}
    prot_x = np.zeros(1000)
    for n, c in enumerate(seq):
        prot_x[n] = token_dict[c]
    # emb
    emb = np.concatenate([np.array(data[n]), np.zeros((1000-len(data[n]),768))], axis=0)
    seq_emb_list.append( [prot_x, emb] )


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

        _prot_data = []
        for n in tqdm(range(self.prot_data.shape[0])):
            pd = self.prot_data[n]
            for seqEmb in seq_emb_list:
                if (pd.reshape(-1) == seqEmb[0]).sum() == 1000:
                    _prot_data.append(seqEmb[1].reshape((1,1000,768)))
                    break
        print(len(_prot_data), self.prot_data.shape[0])
        assert len(_prot_data) == self.prot_data.shape[0]

        self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, self.prot_inter, self.prot_inter_exist, self.label = torch.tensor(self.prot_data), torch.tensor(self.drug_data_ver).float().float(), torch.tensor(self.drug_data_adj).float(), torch.tensor(self.prot_contacts).float(), torch.tensor(self.prot_inter).float(), torch.tensor(self.prot_inter_exist).float().squeeze().float(), torch.tensor(self.label).float()
        self.prot_data = torch.tensor(np.concatenate(_prot_data, axis=0), dtype=torch.float32)
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
            torch.save(model.module.state_dict(), './weights/transformer_yuanfei_' + args.transformer_config + '.pth')

del train_loader
del val_loader


###### evaluation ######
# evaluation
model = net_crossInteraction(args.l0, args.l1, args.l2, args.l3).cuda()
model.load_state_dict(torch.load('./weights/transformer_yuanfei_' + args.transformer_config + '.pth'))
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
with open('./logs/hpo_crossInteraction_parallel.log', 'a+') as f:
    f.write(str(args.l0) + ' ' + str(args.l1) + ' ' + str(args.l2) + ' ' + str(args.l3) + ' ')
eval_set = dataset('val')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader, logging=True, logpath='./logs/hpo_crossInteraction_parallel.log')
prot_length = np.load(data_processed_dir+'prot_dev_length.npy')
comp_length = np.load(data_processed_dir+'comp_dev_length.npy')
cal_interaction_torch(model, eval_loader, prot_length, comp_length, logging=True, logpath='./logs/hpo_crossInteraction_parallel.log')
with open('./logs/hpo_crossInteraction_parallel.log', 'a+') as f:
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


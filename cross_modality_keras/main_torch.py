import torch
import torch.nn as nn

from utils import *
import pdb


class net_crossInteraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.aminoAcid_embedding = nn.Embedding(24, 256)
        self.gru0 = nn.GRU(256, 256, batch_first=True)
        self.gru1 = nn.GRU(256, 256, batch_first=True)
        self.gat = net_prot_gat()
        self.crossInteraction = crossInteraction()
        self.gcn_comp = net_comp_gcn()

        self.relu = nn.ReLU()
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

        self.lambda_l1, self.lambda_fused, self.lambda_group = 0.001, 0.0001, 0.0001
        self.lambda_bind = 10000

    def forward(self, prot_data, drug_data_ver, drug_data_adj, prot_contacts):
        # protein embedding
        aminoAcid_embedding = self.aminoAcid_embedding(prot_data)

        b, i, j, d = aminoAcid_embedding.size()
        prot_seq_embedding = aminoAcid_embedding.view(b*i, j, d)
        prot_seq_embedding, _ = self.gru0(prot_seq_embedding)
        prot_seq_embedding = aminoAcid_embedding.view(b, i, j, d)
        prot_seq_embedding = prot_seq_embedding.view(b*j, i, d)
        prot_seq_embedding, _ = self.gru1(prot_seq_embedding)
        prot_seq_embedding = aminoAcid_embedding.view(b, j, i, d)
        prot_seq_embedding = prot_seq_embedding.view(b, i*j, d)

        prot_graph_embedding = aminoAcid_embedding.view(b, i*j, d)
        prot_graph_embedding = self.gat(prot_graph_embedding, prot_contacts)

        prot_embedding = self.crossInteraction(prot_seq_embedding, prot_graph_embedding)

        # compound embedding
        comp_embedding = self.gcn_comp(drug_data_ver, drug_data_adj)

        # compound-protein interaction
        inter_comp_prot = torch.exp(torch.einsum('bij,bkj->bik', self.joint_attn_prot(self.relu(prot_embedding)), self.joint_attn_comp(self.relu(comp_embedding))))
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
        batch_size, _, _ = inter.size()
        reg_l1 = torch.abs(inter).sum(dim=(1,2)).mean()
        reg_fused = torch.abs(torch.einsum('bij,ti->bjt', inter, fused_matrix)).sum(dim=(1,2)).mean()
        reg_group = ( torch.sqrt(torch.einsum('bij,bki->bjk', inter**2, prot_contacts).sum(dim=1)) * torch.sqrt(prot_contacts.sum(dim=2)) ).sum(dim=1).mean()

        reg_loss = self.lambda_l1 * reg_l1 + self.lambda_fused * reg_fused + self.lambda_group * reg_group
        return reg_loss

    def loss_inter(self, inter, prot_inter, prot_inter_exist):
        label = torch.einsum('b,bij->bij', prot_inter_exist, prot_inter)
        loss = torch.sqrt(((inter - label) ** 2).sum(dim=(1,2))).mean() * self.lambda_bind
        return loss

    def loss_affn(self, affn, label):
        loss = torch.sqrt(((affn - label) ** 2).mean())
        return loss


class net_prot_gat(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.ModuleList([nn.Linear(256, 256) for _ in range(7)])
        self.linear1 = nn.ModuleList([nn.Linear(256, 256) for _ in range(7)])
        self.relu = nn.ReLU()
        self.w_attn = nn.ModuleList([nn.Linear(256, 256) for _ in range(7)])
        self.linear_final = nn.Linear(256, 256)

    def forward(self, x, adj):
        adj[:, list(range(1000)), list(range(1000))] = 1
        for l in range(7):
            x0 = x

            adj_attn = torch.exp(torch.einsum('bij,bkj->bik', self.w_attn[l](x), x))
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


class dataset(torch.utils.data.Dataset):
    def __init__(self, name_split='train'):
        if name_split == 'train':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, _, self.prot_inter, self.prot_inter_exist, self.label = load_train_data('/scratch/user/yuning.you/proj/DeepRelations/HierRNN_GCN_constraint_sup/data_processed/')
        elif name_split == 'val':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, _, self.prot_inter, self.prot_inter_exist, self.label = load_val_data('/scratch/user/yuning.you/proj/DeepRelations/HierRNN_GCN_constraint_sup/data_processed/')
        elif name_split == 'test':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, _, self.prot_inter, self.prot_inter_exist, self.label = load_test_data('/scratch/user/yuning.you/proj/DeepRelations/HierRNN_GCN_constraint_sup/data_processed/')
        elif name_split == 'unseen_prot':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, _, self.prot_inter, self.prot_inter_exist, self.label = load_uniqProtein_data('/scratch/user/yuning.you/proj/DeepRelations/HierRNN_GCN_constraint_sup/data_processed/')
        elif name_split == 'unseen_comp':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, _, self.prot_inter, self.prot_inter_exist, self.label = load_uniqCompound_data('/scratch/user/yuning.you/proj/DeepRelations/HierRNN_GCN_constraint_sup/data_processed/')
        elif name_split == 'unseen_both':
            self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, _, self.prot_inter, self.prot_inter_exist, self.label = load_uniqDouble_data('/scratch/user/yuning.you/proj/DeepRelations/HierRNN_GCN_constraint_sup/data_processed/')

        self.prot_data, self.drug_data_ver, self.drug_data_adj, self.prot_contacts, self.prot_inter, self.prot_inter_exist, self.label = torch.tensor(self.prot_data), torch.tensor(self.drug_data_ver).float().float(), torch.tensor(self.drug_data_adj).float(), torch.tensor(self.prot_contacts).float(), torch.tensor(self.prot_inter).float(), torch.tensor(self.prot_inter_exist).float().squeeze().float(), torch.tensor(self.label).float()
    def __len__(self):
        return self.prot_data.size()[0]
    def __getitem__(self, index):
        return self.prot_data[index], self.drug_data_ver[index], self.drug_data_adj[index], self.prot_contacts[index], self.prot_inter[index], self.prot_inter_exist[index], self.label[index]




train_set = dataset('train')
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=32, shuffle=True)
val_set = dataset('val')
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=32, shuffle=False)
model = net_crossInteraction()
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), 5e-4)

fused_matrix = torch.tensor(np.load('/scratch/user/yuning.you/proj/DeepRelations/HierRNN_GCN_constraint_sup/data_processed/fused_matrix.npy')).cuda()
loss_val_best = 1e10
for epoch in range(10):
    loss_epoch, batch = 0, 0
    for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in train_loader:
        prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label = prot_data.cuda(), drug_data_ver.cuda(), drug_data_adj.cuda(), prot_contacts.cuda(), prot_inter.cuda(), prot_inter_exist.cuda(), label.cuda()

        optimizer.zero_grad()
        inter, affn = model(prot_data, drug_data_ver, drug_data_adj, prot_contacts)
        loss0, loss1, loss2 = model.loss_reg(inter, fused_matrix, prot_contacts), model.loss_inter(inter, prot_inter, prot_inter_exist), model.loss_affn(affn, label)
        loss = loss0 + loss1 + loss2
        print('epoch', epoch, 'batch', batch, loss0.detach().cpu().numpy(), loss1.detach().cpu().numpy(), loss2.detach().cpu().numpy())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        loss_epoch += loss.detach().cpu().numpy()
        batch += 1

    loss_epoch_val, batch_val = 0, 0
    for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in val_loader:
        prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label = prot_data.cuda(), drug_data_ver.cuda(), drug_data_adj.cuda(), prot_contacts.cuda(), prot_inter.cuda(), prot_inter_exist.cuda(), label.cuda()
        with torch.no_grad():
            inter, affn = model(prot_data, drug_data_ver, drug_data_adj, prot_contacts)
            loss0, loss1, loss2 = model.loss_reg(inter, fused_matrix, prot_contacts), model.loss_inter(inter, prot_inter, prot_inter_exist), model.loss_affn(affn, label)
            loss = loss0 + loss1 + loss2
        loss_epoch_val += loss.detach().cpu().numpy()
        batch_val += 1

    print('epoch', epoch, 'train loss', loss_epoch/batch, 'val loss', loss_epoch_val/batch_val)
    if loss_epoch_val/batch_val < loss_val_best:
        loss_val_best = loss_epoch_val/batch_val
        torch.save(model.state_dict(), './crossInteraction_torch.pth')


'''
model.load_state_dict(torch.load('./crossInteraction_torch.pth'))
model = model.cpu()
model.eval()

test_set = dataset('test')
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1e5, shuffle=False)
unseen_prot_set = dataset('unseen_prot')
unseen_prot_loader = torch.utils.data.DataLoader(dataset=unseen_prot_set, batch_size=1e5, shuffle=False)
unseen_comp_set = dataset('unseen_comp')
unseen_comp_loader = torch.utils.data.DataLoader(dataset=unseen_comp_set, batch_size=1e5, shuffle=False)
unseen_both_set = dataset('unseen_both')
unseen_both_loader = torch.utils.data.DataLoader(dataset=unseen_both_set, batch_size=1e5, shuffle=False)

for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in test_loader:
    with torch.no_grad():
        inter, affn = model(prot_data, drug_data_ver, drug_data_adj, prot_contacts)
'''






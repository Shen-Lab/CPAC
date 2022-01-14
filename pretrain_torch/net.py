import torch
import torch.nn as nn
import numpy as np
import pdb


###### define network ######
class net_crossInteraction(nn.Module):
    def __init__(self, p_seq_mask, p_graph_mask, p_seq_replace, p_seq_sub, p_graph_drop, p_graph_sub, lambda_l1=0, lambda_fused=0, lambda_group=0, lambda_bind=0):
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

        self.p_seq_mask, self.p_graph_mask, self.p_seq_replace, self.p_seq_sub, self.p_graph_drop, self.p_graph_sub = p_seq_mask, p_graph_mask, p_seq_replace, p_seq_sub, p_graph_drop, p_graph_sub

    def forward(self, prot_data, drug_data_ver, drug_data_adj, prot_contacts):
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
        affn_comp_prot = affn_comp_prot.reshape(b, 64*32)
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

    def prot_mask(self, prot_data, prot_contacts, prot_len, p_mask=0.15):
        b, i, j = prot_data.size()
        prot_data = prot_data.reshape(b, i*j)

        idx_mask = np.array([[n, m] for n in range(b) for m in np.random.choice(prot_len[n].item(), int(prot_len[n].item()*p_mask), replace=False)])
        label_mask = prot_data[idx_mask[:,0], idx_mask[:,1]]
        prot_data[idx_mask[:,0], idx_mask[:,1]] = 0
        prot_data = prot_data.reshape(b, i, j)
        return prot_data, idx_mask, label_mask

    def forward_seq_mask(self, prot_data, prot_contacts, prot_len):
        prot_data, idx_mask, label_mask = self.prot_mask(prot_data, prot_contacts, prot_len, self.p_seq_mask)

        # protein embedding
        aminoAcid_embedding = self.aminoAcid_embedding(prot_data)

        self.gru0.flatten_parameters()
        self.gru1.flatten_parameters()

        b, i, j, d = aminoAcid_embedding.size()
        prot_seq_embedding = aminoAcid_embedding.reshape(b*i, j, d)
        prot_seq_embedding, _ = self.gru0(prot_seq_embedding)
        prot_seq_embedding = prot_seq_embedding.reshape(b*j, i, d)
        prot_seq_embedding, _ = self.gru1(prot_seq_embedding)
        prot_seq_embedding = prot_seq_embedding.reshape(b, i*j, d)

        '''
        prot_graph_embedding = aminoAcid_embedding.reshape(b, i*j, d)
        prot_graph_embedding = self.gat(prot_graph_embedding, prot_contacts)

        prot_embedding = self.crossInteraction(prot_seq_embedding, prot_graph_embedding)
        '''
        return prot_seq_embedding, idx_mask, label_mask

    def forward_graph_mask(self, prot_data, prot_contacts, prot_len):
        prot_data, idx_mask, label_mask = self.prot_mask(prot_data, prot_contacts, prot_len, self.p_graph_mask)

        # protein embedding
        aminoAcid_embedding = self.aminoAcid_embedding(prot_data)

        '''
        self.gru0.flatten_parameters()
        self.gru1.flatten_parameters()
        '''
        b, i, j, d = aminoAcid_embedding.size()
        '''
        prot_seq_embedding = aminoAcid_embedding.reshape(b*i, j, d)
        prot_seq_embedding, _ = self.gru0(prot_seq_embedding)
        prot_seq_embedding = prot_seq_embedding.reshape(b*j, i, d)
        prot_seq_embedding, _ = self.gru1(prot_seq_embedding)
        prot_seq_embedding = prot_seq_embedding.reshape(b, i*j, d)
        '''

        prot_graph_embedding = aminoAcid_embedding.reshape(b, i*j, d)
        prot_graph_embedding = self.gat(prot_graph_embedding, prot_contacts)

        # prot_embedding = self.crossInteraction(prot_seq_embedding, prot_graph_embedding)
        return prot_graph_embedding, idx_mask, label_mask


    def prot_seq_aug(self, prot_data, prot_contacts, prot_len, p_replace=0.15, p_sub=0.3):
        b, i, j = prot_data.size()
        prot_data = prot_data.reshape(b, i*j)
        mat_mask = torch.zeros((b, i*j), dtype=torch.float32).cuda()
        if np.random.rand(1) <= 0.5:
            # dictionary replacement
            idx_replace = np.array([[n, m] for n in range(b)
                                           for m in np.random.choice(prot_len[n].item(), int(prot_len[n].item()*p_replace), replace=False)])
            token_replace = torch.tensor(np.random.choice(25, idx_replace.shape[0], replace=True)+4, dtype=torch.int64).cuda()

            prot_data[idx_replace[:,0], idx_replace[:,1]] = token_replace
            for n in range(b):
                mat_mask[n, :prot_len[n].item()] = 1

        else:
            # subsequence
            idx_subcentre = np.array([np.random.choice(int(prot_len[n].item()*(1-p_sub))+1, 1)[0]+int(prot_len[n].item()*p_sub*0.5) for n in range(b)])
            idx_sub = np.array([[n, m] for n in range(b)
                                       for m in range(int(idx_subcentre[n]-prot_len[n].item()*p_sub*0.5), int(idx_subcentre[n]+prot_len[n].item()*p_sub*0.5)+1)])

            _prot_data = torch.zeros(prot_data.size(), dtype=torch.int64).cuda()
            _prot_data[idx_sub[:,0], idx_sub[:,1]] = prot_data[idx_sub[:,0], idx_sub[:,1]]
            prot_data = _prot_data
            mat_mask[idx_sub[:,0], idx_sub[:,1]] = 1

        prot_data = prot_data.reshape(b, i, j)
        return prot_data, mat_mask

    def forward_seq_aug(self, prot_data, prot_contacts, prot_len):
        prot_data, mat_mask = self.prot_seq_aug(prot_data, prot_contacts, prot_len, self.p_seq_replace, self.p_seq_sub)

        # protein embedding
        aminoAcid_embedding = self.aminoAcid_embedding(prot_data)

        self.gru0.flatten_parameters()
        self.gru1.flatten_parameters()

        b, i, j, d = aminoAcid_embedding.size()
        prot_seq_embedding = aminoAcid_embedding.reshape(b*i, j, d)
        prot_seq_embedding, _ = self.gru0(prot_seq_embedding)
        prot_seq_embedding = prot_seq_embedding.reshape(b*j, i, d)
        prot_seq_embedding, _ = self.gru1(prot_seq_embedding)
        prot_seq_embedding = prot_seq_embedding.reshape(b, i*j, d)

        '''
        prot_graph_embedding = aminoAcid_embedding.reshape(b, i*j, d)
        prot_graph_embedding = self.gat(prot_graph_embedding, prot_contacts)

        prot_embedding = self.crossInteraction(prot_seq_embedding, prot_graph_embedding)
        '''
        return prot_seq_embedding, mat_mask

    def prot_graph_aug(self, prot_data, prot_contacts, prot_len, p_drop=0.15, p_sub=0.3):
        b, i, j = prot_data.size()
        mat_mask = torch.zeros((b, i*j)).cuda()
        if np.random.rand(1) <= 0.5:
            # node dropping
            for n in range(b):
                idx_perm = np.random.permutation(prot_len[n].item())
                idx_drop = idx_perm[:int(prot_len[n].item()*p_drop)]
                idx_nondrop = idx_perm[int(prot_len[n].item()*p_drop):]

                mat_mask[n, idx_nondrop] = 1
                prot_contacts[n][idx_drop, idx_drop] = 0
                prot_contacts[n][list(range(600)), list(range(600))] = 1
        else:
            # subgraph
            for n in range(b):
                idx_sub = [np.random.choice(prot_len[n].item(), 1)[0]]
                while len(idx_sub) < int(prot_len[n].item()*p_sub):
                    idx_nei = prot_contacts[n][idx_sub].nonzero()[:, 1].cpu().numpy().tolist()
                    idx_nei = [i for i in idx_nei if not i in idx_sub]
                    if idx_nei == []:
                        break
                    idx_sub += idx_nei

                idx_sub = np.array(idx_sub)
                idx_nonsub = np.array([i for i in range(prot_len[n].item()) if not i in idx_sub])

                mat_mask[n, idx_sub] = 1
                prot_contacts[n][idx_nonsub, idx_nonsub] = 0
                prot_contacts[n][list(range(600)), list(range(600))] = 1
        return prot_contacts, mat_mask

    def forward_graph_aug(self, prot_data, prot_contacts, prot_len):
        prot_contacts, mat_mask = self.prot_graph_aug(prot_data, prot_contacts, prot_len, self.p_graph_drop, self.p_graph_sub)

        # protein embedding
        aminoAcid_embedding = self.aminoAcid_embedding(prot_data)

        '''
        self.gru0.flatten_parameters()
        self.gru1.flatten_parameters()
        '''
        b, i, j, d = aminoAcid_embedding.size()
        '''
        prot_seq_embedding = aminoAcid_embedding.reshape(b*i, j, d)
        prot_seq_embedding, _ = self.gru0(prot_seq_embedding)
        prot_seq_embedding = prot_seq_embedding.reshape(b*j, i, d)
        prot_seq_embedding, _ = self.gru1(prot_seq_embedding)
        prot_seq_embedding = prot_seq_embedding.reshape(b, i*j, d)
        '''

        prot_graph_embedding = aminoAcid_embedding.reshape(b, i*j, d)
        prot_graph_embedding = self.gat(prot_graph_embedding, prot_contacts)

        # prot_embedding = self.crossInteraction(prot_seq_embedding, prot_graph_embedding)
        return prot_graph_embedding, mat_mask

    def forward_supervised(self, prot_data, prot_contacts, prot_len, CI=True):
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

        prot_embedding = self.crossInteraction(prot_seq_embedding, prot_graph_embedding, CI=CI)

        return prot_embedding


class net_mask_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 29)
        self.crossentropyloss = nn.CrossEntropyLoss()
    def forward(self, output, idx_mask, label_mask):
        output = self.linear(output)
        loss = self.crossentropyloss(output[idx_mask[:,0], idx_mask[:,1]], label_mask)
        return loss


class net_aug_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection_head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
    def forward(self, output1, output2, mat_mask1, mat_mask2):
        output1 = self.projection_head(torch.einsum('bij,bi,b->bj', output1, mat_mask1, 1/mat_mask1.sum(dim=1)))
        output2 = self.projection_head(torch.einsum('bij,bi,b->bj', output2, mat_mask2, 1/mat_mask2.sum(dim=1)))

        temp = 0.1
        norm1, norm2 = output1.norm(dim=1), output2.norm(dim=1)
        mat_norm = torch.einsum('i,j->ij', norm1, norm2)
        mat_sim = torch.exp(torch.einsum('ik,jk,ij->ij', output1, output2, 1/mat_norm) / temp)

        b, _ = output1.size()
        loss = - torch.log(mat_sim[list(range(b)), list(range(b))] / (mat_sim.sum(dim=1) - mat_sim[list(range(b)), list(range(b))])).mean()
        return loss


class net_supervised_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = np.load('/scratch/user/yuning.you/dataset/pfam_dataset/graph_fc.npy')
        self.classifier_clan = nn.Linear(256, self.fc.shape[0])
        self.classifier_family = nn.ModuleList([nn.Linear(256, f) for f in self.fc])
        self.loss = nn.CrossEntropyLoss(reduction='sum')
    def forward(self, output, label_clan, label_family):
        output = output.mean(dim=1)
        output_clan = self.classifier_clan(output)
        loss_clan = self.loss(self.classifier_clan(output), label_clan)

        loss_family, count = 0, 0
        for n in range(output.size()[0]):
            if self.fc[label_clan[n].item()] > 1:
                loss_family += self.loss(self.classifier_family[label_clan[n]](output[n]).view(1, -1), label_family[n].view(-1))
                count += 1

        loss = (loss_clan + loss_family) / (output.size()[0] + count)
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

    def forward(self, x_seq, x_graph, CI=True):
        if CI:
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


# top model
class net_pretrain(nn.Module):
    def __init__(self, ss_seq, ss_graph, p_seq_mask, p_graph_mask, p_seq_replace, p_seq_sub, p_graph_drop, p_graph_sub):
        super().__init__()
        self.model = net_crossInteraction(p_seq_mask, p_graph_mask, p_seq_replace, p_seq_sub, p_graph_drop, p_graph_sub)
        self.ss_seq, self.ss_graph = ss_seq, ss_graph
        if self.ss_seq == 'mask':
            self.seq_mask_loss = net_mask_loss()
        elif self.ss_seq == 'aug':
            self.seq_aug_loss = net_aug_loss()

        if self.ss_graph == 'mask':
            self.graph_mask_loss = net_mask_loss()
        elif self.ss_graph == 'aug':
            self.graph_aug_loss = net_aug_loss()

    def forward(self, prot_data, prot_contacts, prot_len):
        loss = 0
        if self.ss_seq == 'mask':
            prot_data1 = prot_data.clone()
            prot_seq_embedding, idx_mask, label_mask = self.model.forward_seq_mask(prot_data1, prot_contacts, prot_len)
            loss += self.seq_mask_loss(prot_seq_embedding, idx_mask, label_mask)
        elif self.ss_seq == 'aug':
            prot_data1, prot_data2 = prot_data.detach().clone(), prot_data.detach().clone()
            prot_seq_embedding1, mat_mask1 = self.model.forward_seq_aug(prot_data1, prot_contacts, prot_len)
            prot_seq_embedding2, mat_mask2 = self.model.forward_seq_aug(prot_data2, prot_contacts, prot_len)
            loss += self.seq_aug_loss(prot_seq_embedding1, prot_seq_embedding2, mat_mask1, mat_mask2)

        if self.ss_graph == 'mask':
            prot_data2 = prot_data.clone()
            prot_graph_embedding, idx_mask, label_mask = self.model.forward_graph_mask(prot_data2, prot_contacts, prot_len)
            loss += self.graph_mask_loss(prot_graph_embedding, idx_mask, label_mask)
        elif self.ss_graph == 'aug':
            prot_contacts1, prot_contacts2 = prot_contacts.detach().clone(), prot_contacts.detach().clone()
            prot_graph_embedding1, mat_mask1 = self.model.forward_graph_aug(prot_data, prot_contacts1, prot_len)
            prot_graph_embedding2, mat_mask2 = self.model.forward_graph_aug(prot_data, prot_contacts2, prot_len)
            loss += self.graph_aug_loss(prot_graph_embedding1, prot_graph_embedding2, mat_mask1, mat_mask2)

        return loss


class net_pretrain_supervised(nn.Module):
    def __init__(self, ss_seq, ss_graph, p_seq_mask, p_graph_mask, p_seq_replace, p_seq_sub, p_graph_drop, p_graph_sub, CI=True):
        super().__init__()
        self.model = net_crossInteraction(p_seq_mask, p_graph_mask, p_seq_replace, p_seq_sub, p_graph_drop, p_graph_sub)
        self.ss_supervised_loss = net_supervised_loss()
        self.CI = CI
    def forward(self, prot_data, prot_contacts, prot_len, label_clan, label_family):
        output = self.model.forward_supervised(prot_data, prot_contacts, prot_len, self.CI)
        loss = self.ss_supervised_loss(output, label_clan, label_family)
        return loss

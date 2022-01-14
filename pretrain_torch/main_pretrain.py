import torch
import torch.nn as nn

from utils import *
import pdb
import argparse
import random
from tqdm import tqdm
from net import *


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--data_processed_dir', type=str, default='/scratch/user/yuning.you/proj/DeepRelations/HierRNN_GCN_constraint_sup/data_processed/')
parser.add_argument('--data_pretrain_dir', type=str, default='../pretrain_data/')

parser.add_argument('--ss_seq', type=str, default='mask') # none, mask, aug
parser.add_argument('--ss_graph', type=str, default='mask')
parser.add_argument('--p_seq_mask', type=float, default=0.15)
parser.add_argument('--p_graph_mask', type=float, default=0.15)
parser.add_argument('--p_seq_replace', type=float, default=0.15)
parser.add_argument('--p_seq_sub', type=float, default=0.3)
parser.add_argument('--p_graph_drop', type=float, default=0.15)
parser.add_argument('--p_graph_sub', type=float, default=0.3)

args = parser.parse_args()
print(args)


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
np.random.seed(0)
random.seed(0)


import scipy.sparse as sps
class dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.prot_data, self.prot_contacts, self.prot_len = np.load(args.data_pretrain_dir+'graph_seq.npy'), sps.load_npz(args.data_pretrain_dir+'graph_adj.npz').tocsr(), np.load(args.data_pretrain_dir+'graph_seq_length.npy')
        self.prot_data, self.prot_len = torch.tensor(self.prot_data.reshape((self.prot_data.shape[0], 25, 40))), torch.tensor(self.prot_len)
    def __len__(self):
        return self.prot_data.size()[0]
    def __getitem__(self, index):
        adj = torch.tensor(self.prot_contacts[index].todense().reshape((1000, 1000)))
        for n in range(1, 7):
            adj[list(range(n, 1000)), list(range(1000-n))] = 0
            adj[list(range(1000-n)), list(range(n, 1000))] = 0
        return self.prot_data[index], adj, self.prot_len[index]


###### train ######
pretrain_set = dataset()
pretrain_loader = torch.utils.data.DataLoader(dataset=pretrain_set, batch_size=args.batch_size, shuffle=True)
model = net_pretrain(args.ss_seq, args.ss_graph, args.p_seq_mask, args.p_graph_mask, args.p_seq_replace, args.p_seq_sub, args.p_graph_drop, args.p_graph_sub)
model = nn.DataParallel(model)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

weight_pth = ''
if args.ss_seq == 'none':
    weight_pth += 'seq_none_'
elif args.ss_seq == 'mask':
    weight_pth += ('seq_mask_' + str(args.p_seq_mask) + '_')
elif args.ss_seq == 'aug':
    weight_pth += ('seq_aug_' + str(args.p_seq_replace) + '_' + str(args.p_seq_sub) + '_')
else:
    print('seq ss error')
    assert False

if args.ss_graph == 'none':
    weight_pth += 'graph_none'
elif args.ss_graph == 'mask':
    weight_pth += ('graph_mask_' + str(args.p_graph_mask))
elif args.ss_graph == 'aug':
    weight_pth += ('graph_aug_' + str(args.p_graph_drop) + '_' + str(args.p_graph_sub))
else:
    print('seq ss error')
    assert False

checkpoint_pth = weight_pth + '_chechpoint.pth'
# weight_pth += '.pth'


# resume
start_epoch = 0
if args.resume == 1:
    checkpoint = torch.load('./weights/' + checkpoint_pth)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

import warnings
warnings.filterwarnings("ignore")
loss_epoch_best = 1e10
# pre-train
# for epoch in range(start_epoch, args.epoch):
for epoch in tqdm(range(start_epoch, args.epoch)):
    model.train()
    loss_epoch, batch = 0, 0
    for prot_data, prot_contacts, prot_len in pretrain_loader:
        prot_data, prot_contacts, prot_len = prot_data.cuda(), prot_contacts.cuda(), prot_len.cuda()

        optimizer.zero_grad()
        loss = model(prot_data, prot_contacts, prot_len).mean()
        # print('epoch', epoch, 'batch', batch, 'loss', loss.detach().cpu().numpy())

        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()

        loss_epoch += loss.detach().cpu().numpy()
        batch += 1

    print('epoch', epoch, 'train loss', loss_epoch/batch)
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, './weights/' + checkpoint_pth)
    if loss_epoch/batch < loss_epoch_best:
        print('save epoch ' + str(epoch) + ' weight')
        loss_epoch_best = loss_epoch/batch
        torch.save(model.module.model.state_dict(), './weights/' + weight_pth + '.pth')


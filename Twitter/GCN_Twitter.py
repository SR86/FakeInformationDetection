import sys,os
from Process.rand5fold import load5foldData
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
from tqdm import tqdm
from torch_geometric.nn import GCNConv
import copy
import sys
import os

# 确保 backend 目录在系统路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

class TDrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        if not hasattr(data, 'batch'):
            data.batch = th.zeros(data.x.size(0), dtype=th.long, device=data.x.device)  # 默认每个节点属于一个 batch
        x, edge_index = data.x, data.edge_index
        x1=copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2=copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        x= scatter_mean(x, data.batch, dim=0)

        return x

class BUrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        if not hasattr(data, 'batch'):
            data.batch = th.zeros(data.x.size(0), dtype=th.long, device=data.x.device)  # 默认每个节点属于一个 batch
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x= scatter_mean(x, data.batch, dim=0)
        return x

class Net(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc=th.nn.Linear((out_feats+hid_feats)*2,4)

    def forward(self, data):
        if not hasattr(data, 'batch'):
            data.batch = th.zeros(data.x.size(0), dtype=th.long, device=data.x.device)  # 默认每个节点属于一个 batch
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = th.cat((BU_x,TD_x), 1)
        x=self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

def train_GCN(treeDic, x_test, x_train,TDdroprate,BUdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter):
    model = Net(5000,64,64).to(device)
    BU_params=list(map(id,model.BUrumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    base_params=filter(lambda p:id(p) not in BU_params,model.parameters())
    optimizer = th.optim.Adam([
        {'params':base_params},
        {'params':model.BUrumorGCN.conv1.parameters(),'lr':lr/5},
        {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr/5}
    ], lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate,BUdroprate)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=0)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=0)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            out_labels= model(Batch_data)
            finalloss=F.nll_loss(out_labels,Batch_data.y)
            loss=finalloss
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)



from multiprocessing import freeze_support


def main():
    lr = 0.0005  #学习率
    weight_decay = 1e-4 #权重衰减
    patience = 10 # 早停
    n_epochs = 200 #最大迭代轮数
    batchsize = 128 #每次训练使用128个样本
    TDdroprate = 0.2 #丢弃神经元防止过拟合
    BUdroprate = 0.2 #同上
    datasetname = sys.argv[1]  # "Twitter15"、"Twitter16" #数据据名字
    iterations = int(sys.argv[2])  #迭代次数
    model = "GCN" #模型名称


    test_accs = []
    NR_F1 = []
    FR_F1 = []
    TR_F1 = []
    UR_F1 = []

    for iter in range(iterations):
        fold0_x_test, fold0_x_train, \
        fold1_x_test, fold1_x_train, \
        fold2_x_test, fold2_x_train, \
        fold3_x_test, fold3_x_train, \
        fold4_x_test, fold4_x_train = load5foldData(datasetname)
        treeDic = loadTree(datasetname)

        for i, (x_test, x_train) in enumerate([
            (fold0_x_test, fold0_x_train),
            (fold1_x_test, fold1_x_train),
            (fold2_x_test, fold2_x_train),
            (fold3_x_test, fold3_x_train),
            (fold4_x_test, fold4_x_train),
        ]):
            _, _, _, _, accs, F1, F2, F3, F4 = train_GCN(
                treeDic, x_test, x_train,
                TDdroprate, BUdroprate, lr,
                weight_decay, patience,
                n_epochs, batchsize,
                datasetname, iter)

            test_accs.append(accs)
            NR_F1.append(F1)
            FR_F1.append(F2)
            TR_F1.append(F3)
            UR_F1.append(F4)

    print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
        sum(test_accs) / iterations,
        sum(NR_F1) / iterations,
        sum(FR_F1) / iterations,
        sum(TR_F1) / iterations,
        sum(UR_F1) / iterations))


if __name__ == '__main__':
    freeze_support()
    main()


import pickle
import math
import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import dgl.nn.pytorch as gnn
from EGNN import *
from MUSEAttention import MUSEAttention
import warnings
warnings.filterwarnings("ignore")


# Feature Path
Feature_Path = "./Feature/"
# Seed
SEED = 3407
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

# model parameters
BASE_MODEL_TYPE_1 = 'EGAT'
BASE_MODEL_TYPE_2 = 'EGNN' # agat/gcn
ADD_NODEFEATS = 'all'  # all/atom_feats/psepose_embedding/no

MAP_CUTOFF = 14
DIST_NORM = 15

# INPUT_DIM
if ADD_NODEFEATS == 'all':  # add atom features and psepose embedding
    INPUT_DIM = 54 + 7 + 1
elif ADD_NODEFEATS == 'atom_feats':  # only add atom features
    INPUT_DIM = 54 + 7
elif ADD_NODEFEATS == 'psepose_embedding':  # only add psepose embedding
    INPUT_DIM = 54 + 1
elif ADD_NODEFEATS == 'no':
    INPUT_DIM = 54
HIDDEN_DIM1 = 128 #256  # hidden size of node features
HIDDEN_DIM2 = 256
LAYER = 7  # the number of  layers
DROPOUT = 0.3
ALPHA = 0.7
LAMBDA = 1.5

LEARNING_RATE = 1E-4
WEIGHT_DECAY = 0
BATCH_SIZE = 1
NUM_CLASSES = 2  # [not bind, bind]
NUMBER_EPOCHS = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def embedding(sequence_name):
    pssm_feature = np.load(Feature_Path + "pssm/" + sequence_name + '.npy')
    hmm_feature = np.load(Feature_Path + "hmm/" + sequence_name + '.npy')
    seq_embedding = np.concatenate([pssm_feature, hmm_feature], axis=1)
    return seq_embedding.astype(np.float32)


def get_dssp_features(sequence_name):
    dssp_feature = np.load(Feature_Path + "dssp/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)

def get_res_atom_features(sequence_name):
    res_atom_feature = np.load(Feature_Path + "resAF/" + sequence_name + '.npy')
    return res_atom_feature.astype(np.float32)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def cal_edges(sequence_name, radius=MAP_CUTOFF):  # to get the index of the edges
    dist_matrix = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dist_matrix >= 0) * (dist_matrix <= radius))
    adjacency_matrix = mask.astype(int)
    radius_index_list = np.where(adjacency_matrix == 1)
    radius_index_list = [list(nodes) for nodes in radius_index_list]
    return radius_index_list

def load_graph(sequence_name):
    dismap = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    adjacency_matrix = mask.astype(int)
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return norm_matrix


def graph_collate(samples):
    sequence_name, sequence, label, node_features, G, adj_matrix, pos = map(list, zip(*samples))
    label = torch.Tensor(label)
    G_batch = dgl.batch(G)
    node_features = torch.cat(node_features)
    adj_matrix = torch.Tensor(adj_matrix)
    pos = torch.cat(pos)
    pos = torch.Tensor(pos)
    return sequence_name, sequence, label, node_features, G_batch, adj_matrix, pos

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True):#0.25
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重. 当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.255
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]        [B*N个标签(假设框中有目标)]，[B个标签]
        :return:
        """

        # 固定类别维度，其余合并(总检测框数或总批次数)，preds.size(-1)是最后一个维度
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)

        # 使用log_softmax解决溢出问题，方便交叉熵计算而不用考虑值域
        preds_logsoft = F.log_softmax(preds, dim=1)

        # log_softmax是softmax+log运算，那再exp就算回去了变成softmax
        preds_softmax = torch.exp(preds_logsoft)

        # 这部分实现nll_loss ( crossentropy = log_softmax + nll)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))

        self.alpha = self.alpha.gather(0, labels.view(-1))

        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        # torch.mul 矩阵对应位置相乘，大小一致
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        # torch.t()求转置
        loss = torch.mul(self.alpha, loss.t())
        # print(loss.size()) [1,5]

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss

class ProDataset(Dataset):
    def __init__(self, dataframe, radius=MAP_CUTOFF, dist=DIST_NORM, psepos_path='./Feature/psepos/Train335_psepos_SC.pkl'):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.residue_psepos = pickle.load(open(psepos_path, 'rb'))
        self.radius = radius
        self.dist = dist

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        nodes_num = len(sequence)
        pos = self.residue_psepos[sequence_name]
        reference_res_psepos = pos[0]
        pos = pos - reference_res_psepos
        pos = torch.from_numpy(pos).type(torch.FloatTensor)

        sequence_embedding = embedding(sequence_name)
        structural_features = get_dssp_features(sequence_name)
        node_features = np.concatenate([sequence_embedding, structural_features], axis=1)

        node_features = torch.from_numpy(node_features)
        if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'atom_feats':
            res_atom_features = get_res_atom_features(sequence_name)
            res_atom_features = torch.from_numpy(res_atom_features)
            node_features = torch.cat([node_features, res_atom_features], dim=-1)
        if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'psepose_embedding':
            node_features = torch.cat([node_features, torch.sqrt(
                torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)

        radius_index_list = cal_edges(sequence_name, MAP_CUTOFF)
        edge_feat = self.cal_edge_attr(radius_index_list, pos)

        G = dgl.DGLGraph()
        G.add_nodes(nodes_num)
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = edge_feat.squeeze(1)

        self.add_edges_custom(G,
                              radius_index_list,
                              edge_feat
                              )

        adj_matrix = load_graph(sequence_name)
        node_features = node_features.detach().numpy()
        node_features = node_features[np.newaxis, :, :]
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)

        return sequence_name, sequence, label, node_features, G, adj_matrix, pos

    def __len__(self):
        return len(self.labels)

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2,keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)

        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        radius_attr_list = np.array([distance, cos])
        return radius_attr_list

    def add_edges_custom(self, G, radius_index_list, edge_features):
        src, dst = radius_index_list[1], radius_index_list[0]
        if len(src) != len(dst):
            print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
            raise Exception
        G.add_edges(src, dst)
        G.edata['ex'] = torch.tensor(edge_features)


class EGATBaseModule(nn.Module):
    def __init__(self, in_feats,  edge_feats,out_feats,  num_heads, out_edge_feats,bias):
        super(EGATBaseModule, self).__init__()
        self.EdgeGATConv = gnn.EGATConv(in_feats, edge_feats, out_feats, out_edge_feats, num_heads, bias=True)
       # self.in_features = 2*in_feats
       # self.out_features = out_feats
       # self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        #self.trans = nn.Linear(out_feats * num_heads, out_feats)
        #self.act = nn.ReLU()

       # self.reset_parameters()

    #def reset_parameters(self):
     #   stdv = 1. / math.sqrt(self.out_features)
     #   self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h0, lamda, alpha, l, adj_matrix=None, graph=None, efeats=None):
        theta = min(1, math.log(lamda/l+1))
        if adj_matrix is not None:
            hi = torch.sparse.mm(adj_matrix, input)
        elif graph is not None and efeats is not None:
            hi, efeats = self.EdgeGATConv(graph, input, efeats)
            hi = torch.flatten(hi, start_dim=1, end_dim=2)
            efeats = torch.flatten(efeats, start_dim=1, end_dim=2)
           # hi = self.act(self.trans(hi))
            #hi = F.dropout(hi, 0.1, training=self.training)

        else:
            print('ERROR:adj_matrix, graph and efeats must not be None at the same time! Please input the value of adj_matrix or the value of graph and efeats.')
            raise ValueError
      #  support = torch.cat([hi,h0],1)
       # r = (1-alpha)*hi+alpha*h0
       # output = theta*torch.mm(support, self.weight)+(1-theta)*r
        output = hi+input
        return output, efeats


class EGNNBaseModule(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, edge_feat_size):
        super(EGNNBaseModule, self).__init__()
        self.EGNNConv = EGNN(in_node_nf=in_size, hidden_nf=hidden_size, out_node_nf=out_size, in_edge_nf=edge_feat_size,residual=False,
                             n_layers=1,
                             attention=True)
        #self.in_features = 2*in_size
        #self.out_features = out_size
        #self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))

        #self.reset_parameters()

   # def reset_parameters(self):
     #   stdv = 1. / math.sqrt(self.out_features)
     #   self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, coord_feat, h0, lamda, alpha, l, adj_matrix=None, graph=None, efeats=None):
        theta = min(1, math.log(lamda/l+1))
        if adj_matrix is not None:
            hi = torch.sparse.mm(adj_matrix, input)
        elif graph is not None and efeats is not None:
            hi, pos = self.EGNNConv(input, coord_feat, graph.edges(), efeats)
        else:
            print('ERROR:adj_matrix, graph and efeats must not be None at the same time! Please input the value of adj_matrix or the value of graph and efeats.')
            raise ValueError
      #  support = torch.cat([hi,h0],1)
      #  r = (1-alpha)*hi+alpha*h0
      #  output = theta*torch.mm(support, self.weight)+(1-theta)*r
        output = hi+input
        return output, pos

class EGAT_EGNN(nn.Module):
    def __init__(self, egat_nlayers, egnn_nlayers, nfeat, nhidden1,nhidden2, nclass, dropout, lamda, alpha):
        super(EGAT_EGNN, self).__init__()
        self.egat_baseModules = nn.ModuleList()
        self.egnn_baseModules = nn.ModuleList()
        for _ in range(egat_nlayers):
            self.egat_baseModules.append(EGATBaseModule(in_feats=nhidden1, edge_feats=2, out_feats=nhidden1, out_edge_feats=2, num_heads=1, bias=True))
        for _ in range(egnn_nlayers):
            self.egnn_baseModules.append(EGNNBaseModule(in_size=nhidden2, hidden_size=nhidden2, out_size=nhidden2, edge_feat_size=2))

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden1))
        self.fcs.append(nn.Linear(nfeat, nhidden2))
        self.fcs.append(nn.Linear(egat_nlayers* nhidden1+nhidden2, nhidden1))
        self.fcs.append(nn.Linear(nhidden1, nclass))    ####################################################################################
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.sa = MUSEAttention(d_model=egat_nlayers* nhidden1+nhidden2, d_k=egat_nlayers* nhidden1+nhidden2, d_v=egat_nlayers* nhidden1+nhidden2, h=1)


    def forward(self, x, graph=None, efeats=None, adj_matrix=None, pos=None, type_1=None, type_2=None):
        _layers = []
        _layers2 = []
        #x = F.dropout(x, self.dropout, training=self.training)
        egat_layer_inner = F.dropout(self.act_fn(self.fcs[0](x)), self.dropout, training=self.training)
        egnn_layer_inner = F.dropout(self.act_fn(self.fcs[1](x)), self.dropout, training=self.training)

        egat_out =[]
        _layers.append(egat_layer_inner)
        for i, egat_baseMod in enumerate(self.egat_baseModules):
            if type_1 == 'EGAT':
                egat_layer_inner, efeats = egat_baseMod(input=egat_layer_inner, h0=_layers[0], lamda=self.lamda,
                                                alpha=self.alpha, l=i + 1, graph=graph, efeats=efeats)
            else:
                egat_layer_inner = egat_baseMod(input=egat_layer_inner, h0=_layers[0], lamda=self.lamda,
                                                alpha=self.alpha, l=i + 1, adj_matrix=adj_matrix)

            #egat_layer_inner = self.act_fn(egat_layer_inner)
            egat_layer_inner = F.dropout(egat_layer_inner, self.dropout, training=self.training)
            egat_out.append(egat_layer_inner)

        egat_out = torch.cat(egat_out, dim=1)
        egat_layer_inner = egat_out
        #egat_layer_inner = F.dropout(egat_layer_inner, self.dropout, training=self.training)
        # print("egat")
        # print(egat_layer_inner)


        _layers2.append(egnn_layer_inner)

        for i,egnn_baseMod in enumerate(self.egnn_baseModules):
            if type_2 == 'EGNN':
                egnn_layer_inner, pos = egnn_baseMod(input=egnn_layer_inner, coord_feat=pos, h0=_layers2[0], lamda=self.lamda, alpha=self.alpha, l=i+1, graph=graph, efeats=efeats)
            else:
                egnn_layer_inner = egnn_baseMod(input=egnn_layer_inner, h0=_layers2[0], lamda=self.lamda, alpha=self.alpha, l=i+1, adj_matrix=adj_matrix)

            #egnn_layer_inner = self.act_fn(egnn_layer_inner)
            egnn_layer_inner = F.dropout(egnn_layer_inner, self.dropout, training=self.training)


            #print("egnn")
            #print(egnn_layer_inner)

        layer_inner = torch.cat([egat_layer_inner, egnn_layer_inner], dim=1)
        layer_inner = torch.unsqueeze(layer_inner,dim=0)
        layer_inner = self.sa(layer_inner, layer_inner, layer_inner)
        layer_inner = torch.squeeze(layer_inner, dim=0)
        #layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        #layer_inner = self.act_fn(layer_inner)
        layer_inner = self.fcs[-2](layer_inner)
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.act_fn(layer_inner)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner


class MGMAPPIS(nn.Module):
    def __init__(self,  nfeat, nhidden1,nhidden2, nclass, dropout, lamda, alpha):
        super(MGMAPPIS, self).__init__()

        self.deep_agat = EGAT_EGNN(egat_nlayers=5, egnn_nlayers=7, nfeat=nfeat, nhidden1=nhidden1,nhidden2=nhidden2, nclass=nclass,
                                  dropout=dropout, lamda=lamda, alpha=alpha)
        self.criterion = focal_loss()  # automatically do softmax to the predicted value and one-hot to the label
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.3, patience=5, min_lr=1e-6)

    def forward(self, x, graph, adj_matrix, pos):
        x = x.float()
        x = x.view([x.shape[0]*x.shape[1], x.shape[2]])
        output = self.deep_agat(x=x, graph=graph, efeats=graph.edata['ex'], adj_matrix=adj_matrix, pos=pos, type_1=BASE_MODEL_TYPE_1, type_2=BASE_MODEL_TYPE_2)


        return output

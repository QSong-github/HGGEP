from torch_geometric.nn import TransformerConv, LayerNorm, GATConv, GCNConv
from torch_geometric.utils import to_undirected
# from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch.nn import init
from torch.autograd import Variable
from torch.nn.modules.module import Module
import math
from torch import FloatTensor
from torch.nn.parameter import Parameter
from graph_construction import calcADJ
import time
import random
import numpy as np
from collections import defaultdict

class gs_block(nn.Module):
    def __init__(
        self, feature_dim, embed_dim,
        policy='mean', gcn=True, num_sample=10
    ):
        super().__init__()
        self.gcn = gcn
        self.policy = policy
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.num_sample = num_sample
        self.weight = nn.Parameter(torch.FloatTensor(
            embed_dim,
            self.feat_dim if self.gcn else 2*self.feat_dim
        ))
        init.xavier_uniform_(self.weight)

    def forward(self, x, XY_Adj):
        # x : [315, 1024], 计算每一个[1, 1024]之间的距离
        # move x to cpu first
        # x_np = x.cpu().detach().numpy()
        # # XY_Adj = XY_Adj.cpu().detach().numpy()

        # Dist_adj = calcADJ(x, 4, "euclidean").cpu().detach().numpy()
        # Dist_adj = calculate_distance_matrix(x_np, reverse=True)
        # Dist_adj = calcADJ(x, 0, 'euclidean', all_conn = True)
        # Cos_adj  = calcADJ(x, 5, "cosine").cpu().detach().numpy()

        # norm each adj
        # print("XY_Adj",XY_Adj.shape, XY_Adj[0][:10])
        # print("Dist_adj",Dist_adj.shape, Dist_adj[0][:10])
        # print("Cos_adj",Cos_adj.shape, Cos_adj[0][:10])
        '''
        XY_Adj torch.Size([295, 295]) tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
        Dist_adj torch.Size([295, 295]) tensor([0., 1., 0., 1., 1., 1., 0., 0., 0., 0.])
        Cos_adj torch.Size([295, 295]) tensor([0., 1., 0., 1., 1., 1., 0., 0., 0., 0.])
        '''
        # add
        # a = 0.5
        # Adj = Dist_adj
        # Adj = a * np.array(XY_Adj) + (1-a) * np.array(Dist_adj)
        # a , b = 0.3, 0.3
        # Adj = a * np.array(XY_Adj) + b * np.array(Dist_adj) + (1-a-b) * np.array(Cos_adj)

        # move to cuda
        # x = torch.Tensor(np.array(x)).cuda()
        # Adj = torch.Tensor(np.array(Dist_adj)).cuda()

        # neigh_feats = self.aggregate(x, dj)
        neigh_feats = self.aggregate(x, XY_Adj)

        if not self.gcn:
            combined = torch.cat([x, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.T)).T
        combined = F.normalize(combined, 2, 1)
        return combined

    def aggregate(self, x, Adj):
        adj = Variable(Adj).to(Adj.device)
        if not self.gcn:
            n = len(adj)
            adj = adj-torch.eye(n).to(adj.device)
        if self.policy == 'mean':  # True
            num_neigh = adj.sum(1, keepdim=True) # 对 adj 矩阵的每一行进行求和，并将结果保持为一个列向量，即保持维度不变。
            mask = adj.div(num_neigh) # 该操作将矩阵 adj 的每个元素都除以 num_neigh。进行归一化操作
            to_feats = mask.mm(x)  # 重新分配权重
        elif self.policy == 'max':
            indexs = [i.nonzero() for i in adj == 1]
            to_feats = []
            for feat in [x[i.squeeze()] for i in indexs]:
                if len(feat.size()) == 1:
                    to_feats.append(feat.view(1, -1))
                else:
                    to_feats.append(torch.max(feat, 0)[0].view(1, -1))
            to_feats = torch.cat(to_feats, 0)
        return to_feats

class graph_transf_block(nn.Module):
    def __init__(self, in_dim, num_hidden):
        super().__init__()

        # self.Cos_Adj3 = Cos_Adj(in_dim3, in_dim3)
        # graphTransformer Conv
        self.in_dim, self.num_hidden = in_dim, num_hidden
        self.conv1 = TransformerConv(in_dim, num_hidden)
        self.conv4 = TransformerConv(num_hidden, in_dim)
        
        # relu
        self.activate = F.elu

    def forward(self, x, XY_Adj):
        # get edges
        edge_index_temp = sp.coo_matrix(XY_Adj.cpu().detach().numpy())
        values = edge_index_temp.data  # 边上对应权重值weight
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
        edge_index_A = torch.LongTensor(np.array(indices)).cuda()

        # x and edges
        h1 = self.activate(self.conv1(x, edge_index_A))
        # print(h1.shape)
        # h2 = self.conv2(h1, edge_index_A)
        # # print(h2.shape)
        # h3 = self.activate(self.conv3(h2, edge_index_A))
        # print(h3.shape)
        output = self.conv4(h1, edge_index_A)

        return output

class GT_Cos(nn.Module):
    def __init__(self, in_dim, num_hidden):
        super().__init__()

        self.Cos_Adj = Cos_Adj(in_dim, num_hidden)
        # graphTransformer Conv
        self.in_dim, self.num_hidden = in_dim, num_hidden
        self.conv1 = TransformerConv(in_dim, num_hidden)
        self.conv4 = TransformerConv(num_hidden, in_dim)
        
        # relu
        self.activate = F.elu

    def forward(self, x, XY_Adj):
        # get edges 
        # print(XY_Adj.shape) # n x n
        XY_Adj = self.Cos_Adj(x)
        
        # try 不同尺度 和 多度量融合

        edge_index_temp = sp.coo_matrix(XY_Adj.cpu().detach().numpy())
        values = edge_index_temp.data  # 边上对应权重值weight
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
        edge_index_A = torch.LongTensor(np.array(indices)).cuda()

        # x and edges
        h1 = self.activate(self.conv1(x, edge_index_A))
        # print(h1.shape)
        # h2 = self.conv2(h1, edge_index_A)
        # # print(h2.shape)
        # h3 = self.activate(self.conv3(h2, edge_index_A))
        # print(h3.shape)
        output = self.conv4(h1, edge_index_A)

        return output


class Cos_Adj_yuzhi(Module):
    def __init__(self, in_features=1024, out_features=1024):
        super(Cos_Adj, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight0 = Parameter(FloatTensor(in_features, out_features))
        self.weight1 = Parameter(FloatTensor(in_features, out_features))

        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight0)
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, input):
        seq_len = torch.sum(torch.max(torch.abs(input), dim=1)[0]>0, 1)
        # To support batch operations
        soft = nn.Softmax(1)
        # print(input.shape, seq_len) # [1, 384, 512]
        theta = torch.matmul(input, self.weight0)
        # print(theta.shape) # [2, 512, 32]
        phi = torch.matmul(input, self.weight1)
        # print(phi.shape) # [2, 512, 32]
        phi2 = phi.permute(1, 0)
        sim_graph = torch.matmul(theta, phi2)
        # print(sim_graph.shape) # [2, 576, 576]

        theta_norm = torch.norm(theta, p=2, dim=1, keepdim=True)  # B*T*1
        # print(theta_norm.shape) # [2, 576, 1]
        phi_norm = torch.norm(phi, p=2, dim=1, keepdim=True)  # B*T*1
        # print(phi_norm.shape) # [2, 576, 1]
        x_norm_x = theta_norm.matmul(phi_norm.permute(1, 0))
        # print(x_norm_x.shape) # [2, 576, 576]
        sim_graph = sim_graph / (x_norm_x + 1e-20)

        # 节点之间边太多了，过滤一下
        output = torch.zeros_like(sim_graph)
        # for i in range(len(seq_len)):
        #     tmp = sim_graph[i, :seq_len[i]]
        #     adj2 = tmp
        #     # print(adj2.shape)
        #     adj2 = F.threshold(adj2, 0.7, 0)
        #     adj2 = soft(adj2)
        #     # print(adj2.shape)
        #     output[i, :seq_len[i], :seq_len[i]] = adj2
    
        return sim_graph


class graph_transf_block4(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim):
        super().__init__()

        # graphTransformer Conv
        in_dim, num_hidden = 2048, 1024
        self.conv1 = TransformerConv(in_dim, num_hidden)
        self.conv2 = TransformerConv(num_hidden, in_dim)
        self.conv3 = TransformerConv(in_dim, num_hidden)
        self.conv4 = TransformerConv(num_hidden, in_dim)
        
        # relu
        self.activate = F.elu

    def forward(self, x, XY_Adj):
        # get edges
        
        edge_index_temp = sp.coo_matrix(XY_Adj.cpu().detach().numpy())
        values = edge_index_temp.data  # 边上对应权重值weight
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
        edge_index_A = torch.LongTensor(np.array(indices)).cuda()

        # x and edges
        h1 = self.activate(self.conv1(x, edge_index_A))
        # print(h1.shape)
        h2 = self.conv2(h1, edge_index_A)
        # print(h2.shape)
        h3 = self.activate(self.conv3(h2, edge_index_A))
        # print(h3.shape)
        output = self.conv4(h3, edge_index_A)

        return output

class graph_transf_block4_LSTM(nn.Module):
    def __init__(self, in_dim, num_hidden):
        super().__init__()

        # graphTransformer Conv
        self.in_dim, self.num_hidden = in_dim, num_hidden
        self.conv1 = TransformerConv(in_dim, num_hidden, )
        self.conv2 = TransformerConv(num_hidden, in_dim)
        self.conv3 = TransformerConv(in_dim, num_hidden)
        self.conv4 = TransformerConv(num_hidden, in_dim)
        # self.conv5 = TransformerConv(in_dim, num_hidden)
        # self.conv6 = TransformerConv(num_hidden, in_dim)
        # self.conv7 = TransformerConv(in_dim, num_hidden)
        # self.conv8 = TransformerConv(num_hidden, in_dim)
        
        # relu
        self.activate = F.elu

    def forward(self, x, XY_Adj):
        # get edges
        
        edge_index_temp = sp.coo_matrix(XY_Adj.cpu().detach().numpy())
        values = edge_index_temp.data  # 边上对应权重值weight
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
        edge_index_A = torch.LongTensor(np.array(indices)).cuda()

        # x and edges
        h1 = self.activate(self.conv1(x, edge_index_A))
        # print(h1.shape)
        h2 = self.conv2(h1, edge_index_A)
        # print(h2.shape)
        h3 = self.activate(self.conv3(h2, edge_index_A))
        # print(h3.shape)
        h4 = self.conv4(h3, edge_index_A)
        # print(h4.shape)
        return h4
        # return [h2.unsqueeze(0), h4.unsqueeze(0)]

class GAT(nn.Module):
    def __init__(self, in_dim, num_hidden):
        super().__init__()

        # graphTransformer Conv
        self.in_dim, self.num_hidden = in_dim, num_hidden
        self.conv1 = GATConv(in_dim, num_hidden)
        # self.conv2 = TransformerConv(num_hidden, num_hidden*2)
        # self.conv3 = TransformerConv(num_hidden*2, num_hidden)
        self.conv4 = GATConv(num_hidden, in_dim)
        
        # relu
        self.activate = F.elu

    def forward(self, x, XY_Adj):
        # get edges
        # metric
        # Cos_Adj = self.COS(x)
        
        edge_index_temp = sp.coo_matrix(XY_Adj.cpu().detach().numpy())
        values = edge_index_temp.data  # 边上对应权重值weight
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
        edge_index_A = torch.LongTensor(np.array(indices)).cuda()
        # print(edge_index_A.shape)
        edge_index_A = to_undirected(edge_index_A) # 处理成无向图
        # print(edge_index_A.shape)

        # x and edges
        h1 = self.activate(self.conv1(x, edge_index_A)) 
        # print(h1.shape)
        # h2 = self.conv2(h1, edge_index_A)
        # # print(h2.shape)
        # h3 = self.activate(self.conv3(h2, edge_index_A))
        # print(h3.shape)
        output = self.conv4(h1, edge_index_A)

        return output

class Cos_Adj(Module):
    def __init__(self, in_features=1024, out_features=1024):
        super(Cos_Adj, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight0 = Parameter(FloatTensor(in_features, out_features))
        self.weight1 = Parameter(FloatTensor(in_features, out_features))

        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight0)
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, input):
        # seq_len = torch.sum(torch.max(torch.abs(input), dim=1)[0]>0, 1)
        # To support batch operations
        soft = nn.Softmax(1)
        # print(input.shape, seq_len) # [1, 384, 512]
        theta = torch.matmul(input, self.weight0)
        # print(theta.shape) # [2, 512, 32]
        phi = torch.matmul(input, self.weight1)
        # print(phi.shape) # [2, 512, 32]
        phi2 = phi.permute(1, 0)
        sim_graph = torch.matmul(theta, phi2)
        # print(sim_graph.shape) # [2, 576, 576]

        theta_norm = torch.norm(theta, p=2, dim=1, keepdim=True)  # B*T*1
        # print(theta_norm.shape) # [2, 576, 1]
        phi_norm = torch.norm(phi, p=2, dim=1, keepdim=True)  # B*T*1
        # print(phi_norm.shape) # [2, 576, 1]
        x_norm_x = theta_norm.matmul(phi_norm.permute(1, 0))
        # print(x_norm_x.shape) # [2, 576, 576]
        sim_graph = sim_graph / (x_norm_x + 1e-20)

        # 节点之间边太多了，过滤一下
        output = torch.zeros_like(sim_graph)
        # for i in range(len(seq_len)):
        #     tmp = sim_graph[i, :seq_len[i]]
        #     adj2 = tmp
        #     # print(adj2.shape)
        #     adj2 = F.threshold(adj2, 0.7, 0)
        #     adj2 = soft(adj2)
        #     # print(adj2.shape)
        #     output[i, :seq_len[i], :seq_len[i]] = adj2
    
        return sim_graph

class GAT_Mixer(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim):
        super().__init__()

        # graphTransformer Conv
        in_dim, num_hidden = 1024, 512
        # in_dim, num_hidden = 1024, 768

        self.conv1 = GCNConv(in_dim, num_hidden)
        self.conv2 = GCNConv(num_hidden, in_dim)
        self.conv3 = TransformerConv(in_dim, num_hidden)
        self.conv4 = TransformerConv(num_hidden, in_dim)
        self.conv5 = GATConv(in_dim, num_hidden)
        self.conv6 = GATConv(num_hidden, in_dim)
        self.conv7 = GATConv(in_dim, num_hidden)
        self.conv8 = GATConv(num_hidden, in_dim)
        
        # relu
        self.activate = F.elu

    def forward(self, x, XY_Adj):
        # get edges
        
        edge_index_temp = sp.coo_matrix(XY_Adj.cpu().detach().numpy())
        values = edge_index_temp.data  # 边上对应权重值weight
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
        edge_index_A = torch.LongTensor(np.array(indices)).cuda()
        
        edge_index_A = to_undirected(edge_index_A) # 处理成无向图

        # x and edges
        h1 = self.activate(self.conv1(x, edge_index_A))
        # print(h1.shape)
        h2 = self.conv2(h1, edge_index_A)
        # # print(h2.shape)
        h3 = self.activate(self.conv3(h2, edge_index_A))
        # print(h3.shape)
        h4 = self.conv4(h3, edge_index_A)

        h5 = self.activate(self.conv5(h4, edge_index_A))
        # print(h1.shape)
        h6 = self.conv6(h5, edge_index_A)
        # # print(h2.shape)
        h7 = self.activate(self.conv7(h6, edge_index_A))
        # print(h3.shape)
        h8 = self.conv8(h7, edge_index_A)

        return [h2.unsqueeze(0), h4.unsqueeze(0), h6.unsqueeze(0), h8.unsqueeze(0)]


class HGNN_gs_block(nn.Module):
    def __init__(
        self, feature_dim, embed_dim,
        policy='mean', gcn=True):
        super().__init__()
        self.gcn = gcn
        self.policy = policy
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        # self.num_sample = num_sample

        self.gc2_0 = HGNN(feature_dim, embed_dim, embed_dim)
        self.gc2_1 = HGNN(embed_dim, embed_dim, feature_dim)

    def forward(self, x, XY_Adj):
        # x : [315, 1024], 计算每一个[1, 1024]之间的距离
        # move x to cpu first
        # x_np = x.cpu().detach().numpy()
        # # XY_Adj = XY_Adj.cpu().detach().numpy()

        # Dist_adj = calcADJ(x, 4, "euclidean").cpu().detach().numpy()
        # Dist_adj = calculate_distance_matrix(x_np, reverse=True)
        # Cos_adj  = calcADJ(x, 5, "cosine").cpu().detach().numpy()

        # norm each adj
        # print("XY_Adj",XY_Adj.shape, XY_Adj[0][:10])
        # print("Dist_adj",Dist_adj.shape, Dist_adj[0][:10])
        # print("Cos_adj",Cos_adj.shape, Cos_adj[0][:10])
        '''
        XY_Adj torch.Size([295, 295]) tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
        Dist_adj torch.Size([295, 295]) tensor([0., 1., 0., 1., 1., 1., 0., 0., 0., 0.])
        Cos_adj torch.Size([295, 295]) tensor([0., 1., 0., 1., 1., 1., 0., 0., 0., 0.])
        '''
        # add
        # a = 0.5
        # Adj = Dist_adj
        # Adj = a * np.array(XY_Adj) + (1-a) * np.array(Dist_adj)
        # a , b = 0.3, 0.3
        # Adj = a * np.array(XY_Adj) + b * np.array(Dist_adj) + (1-a-b) * np.array(Cos_adj)

        # move to cuda
        # x = torch.Tensor(np.array(x)).cuda()
        # Adj = torch.Tensor(np.array(Dist_adj)).cuda()

        XY_Adj = np.array(XY_Adj.cpu().detach().numpy())
        H9 = construct_H_with_KNN_from_distance(XY_Adj, 9, False, 1)
        # print(H[0])

        # print("#################",(H9.shape))  # (2, 3, 144, 144)
        G9 = generate_G_from_H(H9, variable_weight=False)
        # G15 = generate_G_from_H(H15_all, variable_weight=False)
        # for i in range(len(G3)):
        #     G3[i] = G3[i].A
        # G3 = torch.Tensor(G3).cuda()
        for i in range(len(G9)):
            G9[i] = G9[i].A
        G9 = torch.Tensor(G9).cuda()
        # for i in range(len(G15)):
        #     G15[i] = G15[i].A
        # G15 = torch.Tensor(G15).cuda()
        # print(G.shape) 
        
        # print(adj1.shape)
        # x1 = torch.relu(self.gc1_0(input, G3))
        # x1 = torch.relu(self.gc1_1(x1, G3))
        # print(x.shape, G9.shape)
        # 
        x1 = torch.relu(self.gc2_0(x, G9))
        # print(x1.shape)
        x2 = torch.relu(self.gc2_1(x1, G9))
        # print(x2.shape)

        return [x, x1, x2]

def calculate_distance_matrix(x, reverse=True):
    N, D = x.shape  # Get the number of objects (rows) and feature dimensions (columns)
    distances = np.zeros((N, N))  # Initialize the distance matrix

    # Calculate pairwise Euclidean distances
    for i in range(N):
        for j in range(N):
            distances[i, j] = np.linalg.norm(x[i] - x[j])  # Euclidean distance between row i and row j
    if reverse:
        min_value = np.min(distances)
        max_value = np.max(distances)
        flipped_matrix = max_value - distances + min_value
        return flipped_matrix
    return distances

class SimilarityAdj(Module):
    def __init__(self, in_features, out_features):
        super(SimilarityAdj, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # self.weight0 = Parameter(FloatTensor(in_features, out_features))
        # self.weight1 = Parameter(FloatTensor(in_features, out_features))
        
        # self.gc1_0 = HGNN(in_features, out_features, out_features)
        # self.gc1_1 = HGNN(out_features, out_features, in_features)
        self.gc2_0 = HGNN(in_features, out_features, out_features)
        self.gc2_1 = HGNN(out_features, out_features, in_features)
        # self.gc3_0 = HGNN(in_features, out_features, out_features)
        # self.gc3_1 = HGNN(out_features, out_features, in_features)
        
        self.register_parameter('bias', None)
        # self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / sqrt(self.weight0.size(1))
        nn.init.xavier_uniform_(self.weight0)
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, input):
        seq_len = torch.sum(torch.max(torch.abs(input), dim=2)[0]>0, 1)
        # To support batch operations
        soft = nn.Softmax(1)
        # print(input.shape) # [1, 384, 512]
        theta = torch.matmul(input, self.weight0)
        # print(theta.shape) # [2, 512, 32]
        phi = torch.matmul(input, self.weight1)
        # print(phi.shape) # [2, 512, 32]
        phi2 = phi.permute(0, 2, 1)
        sim_graph = torch.matmul(theta, phi2)
        # print(sim_graph.shape) # [2, 576, 576]

        theta_norm = torch.norm(theta, p=2, dim=2, keepdim=True)  # B*T*1
        # print(theta_norm.shape) # [2, 576, 1]
        phi_norm = torch.norm(phi, p=2, dim=2, keepdim=True)  # B*T*1
        # print(phi_norm.shape) # [2, 576, 1]
        x_norm_x = theta_norm.matmul(phi_norm.permute(0, 2, 1))
        # print(x_norm_x.shape) # [2, 576, 576]
        sim_graph = sim_graph / (x_norm_x + 1e-20)
        
        # print(sim_graph.shape)
        output = torch.zeros_like(sim_graph)
        for i in range(len(seq_len)):
            tmp = sim_graph[i, :seq_len[i], :seq_len[i]]
            adj2 = tmp
            # print(adj2.shape)
            # adj2 = F.threshold(adj2, 0.7, 0)
            adj2 = soft(adj2)
            # print(adj2.shape)
            output[i, :seq_len[i], :seq_len[i]] = adj2
       
        # distance
        input_dis = input.cpu().detach().numpy()
        dis_mats = []
        for i in range(len(input_dis)):
            dis_mat = Eu_dis(input_dis[i])
            dis_mats.append(dis_mat)
        
        dis_mats = torch.Tensor(np.array(dis_mats))
        # print(sim_graph.shape,  dis_mats.shape)
        output1 = torch.zeros_like(dis_mats)
        for i in range(len(seq_len)):
            tmp = dis_mats[i, :seq_len[i], :seq_len[i]]
            adj2 = tmp
            # print(adj2.shape)
            # adj2 = F.threshold(adj2, 0.7, 0)
            adj2 = soft(adj2)
            # print(adj2.shape)
            output1[i, :seq_len[i], :seq_len[i]] = adj2
        # print(output.shape) # [2, 512, 512]
        
        dis_mats = np.array(output1)
        sim_mats = np.array(output.cpu().detach().numpy())

        # print(sim_graph.shape,  dis_mats.shape)
        
        sim_graph = dis_mats + sim_mats
        # sim_graph = dis_mats
        
        H3_all, H9_all, H15_all = [], [], []
        for i in range(len(sim_graph)):
            # print(sim_graph[i].shape)
            # add multi scale
            # H3 = construct_H_with_KNN_from_distance(sim_graph[i], 3, False, 1)
            H9 = construct_H_with_KNN_from_distance(sim_graph[i], 15, False, 1)
            # H15 = construct_H_with_KNN_from_distance(sim_graph[i], 15, False, 1)
            # print(H[0])
            # H3_all.append(H3)
            H9_all.append(H9)
            # H15_all.append(H15)
        
        # print("#################",(np.array(H9_all)).shape) # (2, 3, 144, 144)
        # hypergraph = np.stack((H3_all, H9_all, H15_all), axis=0)
        # print(hypergraph.shape) # [2,576,576]
        # G3 = generate_G_from_H(H3_all, variable_weight=False)
        G9 = generate_G_from_H(H9_all, variable_weight=False)
        # G15 = generate_G_from_H(H15_all, variable_weight=False)
        # for i in range(len(G3)):
        #     G3[i] = G3[i].A
        # G3 = torch.Tensor(G3).cuda()
        for i in range(len(G9)):
            G9[i] = G9[i].A
        G9 = torch.Tensor(G9).cuda()
        # for i in range(len(G15)):
        #     G15[i] = G15[i].A
        # G15 = torch.Tensor(G15).cuda()
        # print(G.shape) 
        
        # print(adj1.shape)
        # x1 = torch.relu(self.gc1_0(input, G3))
        # x1 = torch.relu(self.gc1_1(x1, G3))
        # print(x1.shape)
        # 
        x2 = torch.relu(self.gc2_0(input, G9))
        x2 = torch.relu(self.gc2_1(x2, G9)) 

        # x3 = torch.relu(self.gc3_0(input, G15))
        # x3 = torch.relu(self.gc3_1(x3, G15))

        # x123 = x1 + x2 + x3

        return x2 + input

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class HGNN(nn.Module):
    def __init__(self, in_ch, n_hid, n_class, dropout=0.5, momentum=0.1):
        super(HGNN, self).__init__()
        self.dropout = dropout
        # self.batch_normalzation1 = nn.BatchNorm1d(in_ch, momentum=momentum)
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        # print(x.shape)
        # x = self.batch_normalzation1(x)
        x = F.relu(self.hgc1(x, G))
        # print(x.shape)
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        # print(x.shape)
        return x

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G

def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    # print(dis_mat.shape)
    n_obj = dis_mat.shape[0] # 闁跨喐鏋婚幏鐑芥晸閺傘倖瀚归柨鐔诲Ν绾板瀚归柨鐔告灮閹风兘鏁撻弬銈嗗N_object
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj): # 闁跨喐鏋婚幏鐑芥晸閺傘倖瀚瑰В蹇庣闁跨喐鏋婚幏鐑芥晸閼哄倻顣幏锟�
        dis_mat[center_idx, center_idx] = 0 # 闁跨喓娈曢弬銈嗗闁跨喐鏋婚幏铚傝礋0
        dis_vec = dis_mat[center_idx] # 闁跨喐鏋婚幏宄板闁跨喕濡喊澶嬪闁跨喐鏋婚幏鐑芥晸閺傘倖瀚规惔鏃堟晸閺傘倖瀚归柨鐔告灮閹凤拷
        # print(dis_vec.shape)
        res_vec = list(reversed(np.argsort(dis_vec)))
        nearest_idx = np.array(res_vec).squeeze()
        avg_dis = np.average(dis_vec) # 閸欐牠鏁撻弬銈嗗閸婏拷
        # print("***\n", avg_dis)
        # any闁跨喓娈曢幘鍛闁跨喐鏋婚幏鐑芥晸閺傘倖瀚归柨鐔告灮閹风兘鏁撻弬銈嗗閸忓啴鏁撻弬銈嗗闁跨喐鏋婚幏鐑芥晸閺傘倖瀚归柨鐔告灮閹风兘鏁撴慨鎰剁礉闁跨喐鏋婚幏鐑芥晸閺傘倖瀚筎rue闁跨喐瑙︽潻鏂剧串閹风ǖrue
        if not np.any(nearest_idx[:k_neig] == center_idx):
            # 闁跨喐鏋婚幏鐑芥晸閺傘倖瀚归柨鐔活潡閿燂拷10闁跨喐鏋婚幏宄板帗闁跨喐鏋婚幏鐑芥晸閺傘倖瀚归柨鐔活潡閸氾缚绗夌敮顔藉闁跨喐鏋婚幏鐑芥晸閺傘倖瀚归崜宥夋晸閼哄倻鍋ｉ敍宀勬晸閺傘倖瀚归柨鐔告灮閹风兘鏁撻弬銈嗗闁跨喖鍙洪弬銈嗗
            nearest_idx[k_neig - 1] = center_idx

        # print(nearest_idx[:k_neig])
        for node_idx in nearest_idx[:k_neig]:
            if is_probH: # True, 閸欐牠鏁撻弬銈嗗闁跨喓绮搁敐蹇斿闁跨喐鏋婚幏鐑芥晸閺傘倖瀚归柨鐔告灮閹疯渹绔撮柨鐔告灮閹风兘鏁撻弬銈嗗绾噣鏁撴笟銉ь劜閹风兘鏁撻弬銈嗗
                H[node_idx, center_idx] = np.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
                # print(H[node_idx, center_idx])
            else:
                # print("#")
                H[node_idx, center_idx] = 1.0
        # print(H)
    return H


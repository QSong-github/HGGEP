import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv
from torch_geometric.data import Data
import anndata as ann
import numpy as np
import scanpy as sc
import pytorch_lightning as pl
import torchvision.transforms as tf
from sklearn.cluster import KMeans
from NB_module import *
from transformer import *
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from copy import deepcopy as dcp
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import math
from torchvision import models

class Conv2d_Hori_Veri_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Hori_Veri_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(
            1, 5), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1],
                                self.conv.weight[:, :, :, 2], self.conv.weight[:, :, :, 3], tensor_zeros, self.conv.weight[:, :, :, 4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias,
                              stride=self.conv.stride, padding=1)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias,
                                stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

def CD_ConvBNReLU(in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, CD_Conv=Conv2d_Hori_Veri_Cross):
    return nn.Sequential(
        CD_Conv(in_channels, out_channels, kernel_size,
                stride, padding, theta=0.7, groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )

def ConvBNReLU(in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride, padding, dilation, groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class GEM(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, groups=1):
        super(GEM, self).__init__()
        self.branch1 = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=2, groups=groups)
        self.branch2 = CD_ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=2, groups=groups)
        self.branch_last = ConvBNReLU(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat([out1, out2], dim=1)
        out = self.branch_last(out)
        return out

class HGGEP(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, fig_size=112, label=None,
                 dropout=0.2, n_pos=64, kernel_size=5, patch_size=7, n_genes=785,
                 depth1=2, depth2=8, depth3=4, heads=16, channel=32,
                 zinb=0.25, nb=False, bake=5, lamb=0.5, policy='mean',
                 ):
        super().__init__()
        # self.save_hyperparameters()
        dim = 1024
        self.learning_rate = learning_rate

        self.nb = nb
        self.zinb = zinb
        self.bake = bake
        self.lamb = lamb
        self.label = label
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        self.channel = channel
        self.patch_size = patch_size
        self.n_genes = n_genes

        # first conv
        self.conv0 = GEM(3,3)
        
        # Mobile
        shufflenet_v2 = models.shufflenet_v2_x0_5(pretrained=True)
        self.layer1 = shufflenet_v2.conv1
        self.layer2 = shufflenet_v2.stage2
        self.layer3 = shufflenet_v2.stage3
        self.layer4 = shufflenet_v2.stage4
        self.layer5 = shufflenet_v2.conv5
        
        self.down4 = nn.Sequential(CBAM(192), nn.BatchNorm2d(192), nn.Conv2d(192, 64, 1, 1), nn.Flatten())
        self.down5 = nn.Sequential(CBAM(1024), nn.BatchNorm2d(1024), nn.Conv2d(1024, 64, 1, 1), nn.Flatten())
        self.down6 = nn.Sequential(CBAM(1024), nn.BatchNorm2d(1024), nn.Conv2d(1024, 1024, 1, 1), nn.Flatten())
        
        # ViT Encoder
        self.ViT1 = nn.Sequential(*[attn_block(dim, 16, 64, 1024, dropout) for i in range(4)])
        self.ViT2 = nn.Sequential(*[attn_block(dim, 16, 64, 1024, dropout) for i in range(4)])
        self.ViT3 = nn.Sequential(*[attn_block(dim, 16, 64, 1024, dropout) for i in range(4)])

        self.jknet = nn.Sequential(nn.LSTM(dim, dim, 4), SelectItem(0))

        self.hgnn = HypergraphNeuralNetwork(input_dim=1024, hidden_dim=512, output_dim=1024)

        # head
        self.gene_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))
        
        if self.zinb > 0:  # default = 0.25
            if self.nb:
                self.hr = nn.Linear(dim, n_genes)
                self.hp = nn.Linear(dim, n_genes)
            else:
                self.mean = nn.Sequential(nn.Linear(dim, n_genes), MeanAct())
                self.disp = nn.Sequential(nn.Linear(dim, n_genes), DispAct())
                self.pi = nn.Sequential(nn.Linear(dim, n_genes), nn.Sigmoid())
        if self.bake > 0:  # 5 the number of augmented images.
            self.coef = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
            )

        self.tf = tf.Compose([
            tf.RandomGrayscale(0.1),
            tf.RandomRotation(90),
            tf.RandomHorizontalFlip(0.2)
        ])

    def forward(self, patches, centers, adj, aug=False):
        B, N, C, H, W = patches.shape
        patches = patches.reshape(B*N, C, H, W)  # ([n spots, 3, 112, 112])
 
        patches = self.conv0(patches)
        h3 = self.layer3(self.layer2(self.layer1(patches)))
        # print("h3:",h3.shape)                 # [295, 96, 8, 8]
        h4_tmp = self.layer4(h3)
        # print(h4_tmp.shape)                   # [295, 192, 4, 4]
        h4 = self.down4(h4_tmp)
        # print(h4.shape)                     
        h5_tmp = self.layer5(h4_tmp)
        # print(h5_tmp.shape)                   # [295, 1024, 4, 4]
        h5 = self.down5(h5_tmp)
        # print(h5.shape)                       # [295, 512]
        h6 = self.down6(F.adaptive_max_pool2d(h5_tmp, (1, 1)))
        # print(h6.shape)                       #  [295, 1568]

        # print(patches.shape)  # ([n, 512, 3, 3])
        centers_x = self.x_embed(centers[:, :, 0]) 
        centers_y = self.y_embed(centers[:, :, 1])
        ct = centers_x + centers_y
        h4 = self.ViT1(h4.unsqueeze(0)+ct).squeeze(0)
        h5 = self.ViT2(h5.unsqueeze(0)+ct).squeeze(0)
        h6 = self.ViT3(h6.unsqueeze(0)+ct).squeeze(0)
        # print(h6[0][:20])

        # HGNN
        HGNN_data = build_adj_hypergraph(h4+h5+h6, adj, 3).cuda()
        hgnn = self.hgnn(HGNN_data)
        jk = [h4.unsqueeze(0), h5.unsqueeze(0), h6.unsqueeze(0), hgnn.unsqueeze(0)]
        g = torch.cat(jk, 0)
        h = self.jknet(g).mean(0)

        x = self.gene_head(h)
        extra = None
        if self.zinb > 0:
            if self.nb:
                r = self.hr(h)
                p = self.hp(h)
                extra = (r, p)
            else:
                m = self.mean(h)
                d = self.disp(h)
                p = self.pi(h)
                extra = (m, d, p)
        if aug:
            h = self.coef(h)
        return x, extra, h

    def aug(self, patch, center, adj):
        bake_x = []
        for i in range(self.bake):
            new_patch = self.tf(patch.squeeze(0)).unsqueeze(0)
            x, _, h = self(new_patch, center, adj, True)
            bake_x.append((x.unsqueeze(0), h.unsqueeze(0)))
        return bake_x

    def distillation(self, bake_x):
        new_x, coef = zip(*bake_x)
        coef = torch.cat(coef, 0)
        new_x = torch.cat(new_x, 0)
        coef = F.softmax(coef, dim=0)
        new_x = (new_x*coef).sum(0)
        return new_x

    def training_step(self, batch, batch_idx):
        patch, center, exp, adj, oris, sfs, *_ = batch
        adj = adj.squeeze(0)
        exp = exp.squeeze(0)
        pred, extra, h = self(patch, center, adj)

        mse_loss = F.mse_loss(pred, exp)
        bake_loss = 0
        if self.bake > 0: # the number of augmented images.
            bake_x = self.aug(patch, center, adj)
            new_pred = self.distillation(bake_x)
            bake_loss += F.mse_loss(new_pred, pred)
        zinb_loss = 0
        if self.zinb > 0:
            if self.nb:
                r, p = extra
                zinb_loss = NB_loss(oris.squeeze(0), r, p)
            else:
                m, d, p = extra
                zinb_loss = ZINB_loss(oris.squeeze(0), m, d, p, sfs.squeeze(0))

        loss = mse_loss+self.zinb*zinb_loss+self.lamb*bake_loss
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp, adj, oris, sfs, *_ = batch

        def cluster(pred, cls):
            sc.pp.pca(pred)
            sc.tl.tsne(pred)
            kmeans = KMeans(n_clusters=cls, init="k-means++",
                            random_state=0).fit(pred.obsm['X_pca'])
            pred.obs['kmeans'] = kmeans.labels_.astype(str)
            p = pred.obs['kmeans'].to_numpy()
            return p

        pred, extra, h = self(patch, center, adj.squeeze(0))
        if self.label is not None:
            adata = ann.AnnData(pred.squeeze().cpu().numpy())
            idx = self.label != 'undetermined'
            cls = len(set(self.label))
            x = adata[idx]
            l = self.label[idx]
            predlbl = cluster(x, cls-1)
            # self.log('nmi', nmi_score(predlbl, l))
            # self.log('ari', ari_score(predlbl, l))

        loss = F.mse_loss(pred.squeeze(0), exp.squeeze(0))
        # self.log('valid_loss', loss, on_epoch=True, prog_bar=True, logger=True)

        # pred = pred.squeeze(0).cpu().numpy().T
        # exp = exp.squeeze(0).cpu().numpy().T

        # r = []
        # for g in range(self.n_genes):
        #     r.append(pearsonr(pred[g], exp[g])[0])
        # R = torch.Tensor(r).mean()
        # self.log('R', R, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        StepLR = torch.optim.lr_scheduler.StepLR(
            optim, step_size=50, gamma=0.9)
        optim_dict = {'optimizer': optim, 'lr_scheduler': StepLR}
        return optim_dict

def euclidean_distance(x1, x2):
    return torch.norm(x1 - x2, dim=-1)

def cosine_similarity(x1, x2):
    dot_product = torch.sum(x1 * x2, dim=-1)
    norm_x1 = torch.norm(x1, dim=-1)
    norm_x2 = torch.norm(x2, dim=-1)
    similarity = dot_product / (norm_x1 * norm_x2)
    return similarity

def build_sparse_hypergraph(features, num_neighbors):
    n, m = features.size()

    hypergraph_nodes = torch.arange(n).view(1, -1)

    hypergraph_edges = []
    edge_weights = []

    for i in range(n):
        distances = euclidean_distance(features[i, :], features)

        _, nearest_neighbors = torch.topk(-distances, k=num_neighbors + 1)

        nearest_neighbors = nearest_neighbors[nearest_neighbors != i]

        for neighbor in nearest_neighbors:
            hypergraph_edges.append([i, neighbor])
            edge_weights.append(distances[neighbor])

    hypergraph_edges = torch.tensor(hypergraph_edges, dtype=torch.long).t()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    data = Data(x=features, edge_index=hypergraph_edges, edge_attr=edge_weights, y=None)

    return data

def normalize_tensor(tensor):
    normalized_tensor = torch.div(tensor, tensor.sum())
    
    return normalized_tensor

def build_adj_hypergraph(features, adjacency_matrix, num_neighbors):
    n, m = features.size()

    hypergraph_edges = []
    edge_weights = []

    for i in range(n):
        # Neighbor_distances = adjacency_matrix[i, :]
        Neighbor_distances = normalize_tensor(adjacency_matrix[i, :])
        Eud_distances = normalize_tensor(euclidean_distance(features[i, :], features))
        # print(Neighbor_distances[:10],  Eud_distances[:10], Neighbor_distances_[:10]) # [1,1,0,0]
        
        distances = Neighbor_distances + Eud_distances
        # distances = Neighbor_distances

        _, nearest_neighbors = torch.topk(distances, k=num_neighbors + 1)
        # exit()

        for neighbor in nearest_neighbors:
            hypergraph_edges.append([i, neighbor])
            edge_weights.append(distances[neighbor])

    hypergraph_edges = torch.tensor(hypergraph_edges, dtype=torch.long).t()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    data = Data(x=features, edge_index=hypergraph_edges, edge_attr=edge_weights, y=None)

    return data

def build_hypergraph_from_adjacency(features, adjacency_matrix, threshold=0.0):
    n = adjacency_matrix.size(0)

    hypergraph_nodes = torch.arange(n).view(1, -1)

    hypergraph_edges = []
    edge_weights = []

    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] > threshold:
                if i not in hypergraph_nodes or j not in hypergraph_nodes:
                    continue
                hypergraph_edges.append([i, j])
                edge_weights.append(adjacency_matrix[i, j])

    hypergraph_edges = torch.tensor(hypergraph_edges, dtype=torch.long).t()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    data = Data(x=features, edge_index=hypergraph_edges, edge_attr=edge_weights, y=None)

    return data

class HypergraphNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HypergraphNeuralNetwork, self).__init__()
        self.conv1 = HypergraphConv(input_dim, hidden_dim)
        self.conv2 = HypergraphConv(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, data):
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x1 = F.dropout(self.norm(torch.relu(x)), 0.5)
        x2 = self.conv2(x1, data.edge_index, data.edge_attr)

        return x2 + data.x


if __name__ == "__main__":
    net = shufflenet_v2().cuda()
    out = net(torch.rand(1, 3, 64, 64).cuda())
    print(out.shape)

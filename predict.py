import torch
import numpy as np
import scanpy as sc
import anndata as ad
from tqdm import tqdm
from dataset import ViT_HER2ST, ViT_SKIN, ViT_BDS
from scipy.stats import pearsonr,spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

def pk_load(fold,mode='train',flatten=False,dataset='her2st',r=4,ori=True,adj=True,prune='Grid',neighs=4):
    # assert dataset in ['her2st','cscc']
    if dataset=='her2st':
        dataset = ViT_HER2ST(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    elif dataset=='cscc':
        dataset = ViT_SKIN(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    elif dataset=='BDS':
        dataset = ViT_BDS(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    return dataset

def test(model,test,device='cuda'):
    model=model.to(device)
    model.eval()
    preds=None
    ct=None
    gt=None
    loss=0
    with torch.no_grad():
        for patch, position, exp, adj, *_, center in tqdm(test):
            patch, position, adj = patch.to(device), position.to(device), adj.to(device).squeeze(0)
            pred = model(patch, position, adj)[0]
            preds = pred.squeeze().cpu().numpy()
            ct = center.squeeze().cpu().numpy()
            gt = exp.squeeze().cpu().numpy()
    adata = ad.AnnData(preds)
    adata.obsm['spatial'] = ct
    adata_gt = ad.AnnData(gt)
    adata_gt.obsm['spatial'] = ct
    return adata,adata_gt


def leiden_cluster(adata, label, resolution=1.0):
    idx = label != 'undetermined'
    tmp = adata[idx]
    l = label[idx]
    
    sc.pp.pca(tmp)
    sc.tl.tsne(tmp)

    # Compute neighborhood graph
    sc.pp.neighbors(tmp)
    
    print("Leiden")
    sc.tl.leiden(tmp, key_added='leiden', resolution=resolution)
    p = tmp.obs['leiden'].astype(str)
    
    lbl = np.full(len(adata), str(len(set(l))))
    lbl[idx] = p
    
    adata.obs['x'] = lbl
    
    return p, round(ari_score(p, l), 3)

def cluster(adata, label, method='kmeans', resolution=1.0):
    if method == 'leiden':
        return leiden_cluster(adata, label, resolution=resolution)
    elif method == 'kmeans':
        idx = label != 'undetermined'
        tmp = adata[idx]
        l = label[idx]

        sc.pp.pca(tmp)
        sc.tl.tsne(tmp) 
        kmeans = KMeans(n_clusters=len(set(l)), init="k-means++", random_state=0).fit(tmp.obsm['X_pca'])
        p = kmeans.labels_.astype(str)

        lbl = np.full(len(adata), str(len(set(l))))
        lbl[idx] = p

        adata.obs['x'] = lbl

        return p, round(ari_score(p, l), 3)
    else:
        raise ValueError("Unsupported clustering method. Supported methods: 'leiden', 'kmeans'")

def replace_nans_infs_numpy(array1, array2):
    array1 = np.nan_to_num(array1)
    array2 = np.nan_to_num(array2)
    return array1, array2


def get_R(data1,data2,dim=1,func=pearsonr):
    adata1=data1.X
    adata2=data2.X
    # adata1, adata2 = replace_nans_infs_numpy(adata1, adata2)
    r1,p1=[],[]
    for g in range(data1.shape[dim]):
        if dim==1:
            r,pv=func(adata1[:,g],adata2[:,g])
        elif dim==0:
            r,pv=func(adata1[g,:],adata2[g,:])
        r1.append(r)
        p1.append(pv)
    r1=np.array(r1)
    p1=np.array(p1)
    return r1,p1

import torch
import numpy as np
from scipy.spatial import distance_matrix, minkowski_distance, distance
def calcADJ(coord, k=8, distanceType='euclidean', pruneTag='NA', all_conn = False):
    r"""
    Calculate spatial Matrix directly use X/Y coordinates
    """
    spatialMatrix=coord #.cpu().numpy()
    # print(spatialMatrix.shape)  # (587, 2)
    nodes=spatialMatrix.shape[0]
    Adj=torch.zeros((nodes,nodes))
    # print("adj",Adj.shape) # ([712, 712])
    distMat_all = []
    for i in np.arange(spatialMatrix.shape[0]):
        tmp=spatialMatrix[i,:].reshape(1,-1)
        # calculate euclidean
        # print(tmp.shape, tmp)  # (1, 2) 计算每一个
        # cdist 函数返回一个形状为 (m, k) 的二维数组，其中包含了 XA 中每个数据点与 XB 中每个数据点之间的距离。
        distMat = distance.cdist(tmp,spatialMatrix, 'euclidean')
        # distMat2 = distance.cdist(tmp,spatialMatrix, 'cosine')
        # print(distMat1.shape, distMat1[:10], distMat2.shape, distMat2[:10])  # (1, 712),  (1, 712)
        '''mode
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
        'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
        'sqeuclidean', 'wminkowski', 'yule'.
        '''
        # return all connected graph
        if all_conn:
            distMat_all.append(distMat.squeeze(0))
            continue

        if k == 0:
            k = spatialMatrix.shape[0]-1
            
        # rank k
        res = distMat.argsort()[:k+1]
        tmpdist = distMat[0,res[0][1:k+1]]
        boundary = np.mean(tmpdist)+np.std(tmpdist) #optional
        for j in np.arange(1,k+1):
            # No prune
            if pruneTag == 'NA':
                Adj[i][res[0][j]]=1.0
            elif pruneTag == 'STD':
                if distMat[0,res[0][j]]<=boundary:
                    Adj[i][res[0][j]]=1.0
            # Prune: only use nearest neighbor as exact grid: 6 in cityblock, 8 in euclidean
            elif pruneTag == 'Grid':  # True
                if distMat[0,res[0][j]]<=2.0:
                    Adj[i][res[0][j]]=1.0
    if all_conn:
        distMat_all = torch.Tensor(np.array(distMat_all)).cuda()
        # print(distMat_all.shape)
        return distMat_all
    return Adj
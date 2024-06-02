import torch
import numpy as np
from scipy.spatial import distance_matrix, minkowski_distance, distance
def calcADJ(coord, k=8, distanceType='euclidean', pruneTag='NA', all_conn = False):
    r"""
    Calculate spatial Matrix directly use X/Y coordinates
    """
    spatialMatrix=coord
    nodes=spatialMatrix.shape[0]
    Adj=torch.zeros((nodes,nodes))
    distMat_all = []
    for i in np.arange(spatialMatrix.shape[0]):
        tmp=spatialMatrix[i,:].reshape(1,-1)
        distMat = distance.cdist(tmp,spatialMatrix, 'euclidean')
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
            elif pruneTag == 'Grid':  # True
                if distMat[0,res[0][j]]<=2.0:
                    Adj[i][res[0][j]]=1.0
    if all_conn:
        distMat_all = torch.Tensor(np.array(distMat_all)).cuda()
        return distMat_all
    return Adj
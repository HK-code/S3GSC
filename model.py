import numpy as np
from munkres import Munkres
from scipy.sparse.linalg import svds
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
from scipy.linalg import solve, kron, solve_sylvester
from sklearn.metrics import normalized_mutual_info_score, cohen_kappa_score



class DVSGSC:
    '''
    对视图一致性约束的目标函数求解
    '''

    def __init__(self, n_clusters, spatial_graph, spectral_graph, alpha, beta, gamma=0, n_view=2, max_iter=1000):
        self.n_clusters = n_clusters
        self.spatial_graph = spatial_graph  # spatial_graph -> (n, n)
        self.spectral_graph = spectral_graph  # spectral_graph -> (n, n)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.n_view = n_view

    def __adjacent_mat(self, A):
        """
        Construct normlized adjacent matrix, N.B. consider only connection of k-nearest graph
        :param x: array like: n_sample * n_feature
        :return:
        """
        # A = kneighbors_graph(x, n_neighbors=n_neighbors, include_self=True).toarray()
        A = A * np.transpose(A)
        D = np.diag(np.reshape(np.sum(A, axis=1) ** -0.5, -1))
        normlized_A = np.dot(np.dot(D, A), D)
        return normlized_A

    def unnormalized_laplacian(self, adj_matrix):
        # 先求度矩阵
        R = np.sum(adj_matrix, axis=1)
        degreeMatrix = np.diag(R)
        return degreeMatrix - adj_matrix

    def update_C(self, X, A1, A2, E1, E2):
        I = np.eye(X.shape[1])
        XA_1 = np.dot(X, A1)
        XA_2 = np.dot(X, A2)
        ATXTXA_1 = np.dot(np.transpose(XA_1), XA_1)
        ATXTXA_2 = np.dot(np.transpose(XA_2), XA_2)
        K0 = ATXTXA_1 + ATXTXA_2 + self.beta * I
        K1 = np.dot(np.transpose(XA_1), X) + np.dot(np.transpose(XA_2), X) - np.dot(ATXTXA_1, E1) - np.dot(ATXTXA_2, E2)
        K0_inv = np.linalg.inv(K0)
        C = np.dot(K0_inv, K1)
        return C

    def update_E(self, X, C, A):
        I = np.eye(X.shape[1])
        XA = np.dot(X, A)
        ATXTXA = np.dot(np.transpose(XA), XA)
        K0 = ATXTXA + self.alpha * I
        K1 = np.dot(np.transpose(XA), X) - np.dot(ATXTXA, C)
        K0_inv = np.linalg.inv(K0)
        E = np.dot(K0_inv, K1)
        return E

    def initialize(self, X, A):
        I = np.eye(X.shape[1])
        K0 = np.dot(X, A)  # XA
        K1 = np.dot(np.transpose(K0), K0) + self.alpha * I  # A.TX.TXA+alpha*I
        K2 = np.dot(np.transpose(K0), X)  # A.TX.TX
        K1_inv = np.linalg.inv(K1)
        C = np.dot(K1_inv, K2)
        return C

    def fit(self, X):
        A_spa = self.__adjacent_mat(self.spatial_graph)
        A_spe = self.__adjacent_mat(self.spectral_graph)
        E1 = np.zeros_like(A_spe)
        E2 = np.zeros_like(A_spe)
        Coef = np.zeros_like(A_spe)
        for iter in range(self.max_iter):
            C = self.update_C(X, A_spa, A_spe, E1, E2)
            E1 = self.update_E(X, C, A_spa)
            E2 = self.update_E(X, C, A_spe)
            if np.max(Coef - C) < 0.001:
                break
            else:
                Coef = C
        y_pre, C_final = self.post_proC(Coef, self.n_clusters, 8, 18)
        return y_pre

    def thrC(self, C, ro):
        if ro < 1:
            N = C.shape[1]
            Cp = np.zeros((N, N))
            S = np.abs(np.sort(-np.abs(C), axis=0))
            Ind = np.argsort(-np.abs(C), axis=0)
            for i in range(N):
                cL1 = np.sum(S[:, i]).astype(float)
                stop = False
                csum = 0
                t = 0
                while (stop == False):
                    csum = csum + S[t, i]
                    if csum > ro * cL1:
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C
        return Cp

    def build_aff(self, C):
        N = C.shape[0]
        Cabs = np.abs(C)
        ind = np.argsort(-Cabs, 0)
        for i in range(N):
            Cabs[:, i] = Cabs[:, i] / (Cabs[ind[0, i], i] + 1e-6)
        Cksym = Cabs + Cabs.T
        return Cksym

    def post_proC(self, C, K, d, alpha):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        C = 0.5 * (C + C.T)
        r = d * K + 1
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        # grp = utils.spectral_clustering(L, K, K) + 1
        # print(grp.shape)
        # spectral = SpectralClustering(n_clusters=K, eigen_solver=None, affinity='precomputed',
        #                               assign_labels='discretize')
        spectral = SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                      assign_labels='discretize')
        spectral.fit(L)
        grp = spectral.fit_predict(L) + 1
        # print(grp.shape)
        return grp, L


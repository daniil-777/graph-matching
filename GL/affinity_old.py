import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
import math

from utils.build_graphs import reshape_edge_feature_gl
class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix via inner product from feature space.
    Me = X * Lambda * Y^T
    Mp = Ux * Uy^T
    Parameter: scale of weight d
    Input: edgewise (pairwise) feature X, Y
           pointwise (unary) feature Ux, Uy
    Output: edgewise affinity matrix Me
            pointwise affinity matrix Mp
    Weight: weight matrix Lambda = [[Lambda1, Lambda2],
                                    [Lambda2, Lambda1]]
            where Lambda1, Lambda2 > 0
    """
    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        self.lambda1 = Parameter(Tensor(self.d, self.d))
        self.lambda2 = Parameter(Tensor(self.d, self.d))
        self.relu = nn.ReLU()  # problem: if weight<0, then always grad=0. So this parameter is never updated!
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.lambda1.size(1) * 2)
        self.lambda1.data.uniform_(-stdv, stdv)
        self.lambda2.data.uniform_(-stdv, stdv)
        self.lambda1.data += torch.eye(self.d) / 2
        self.lambda2.data += torch.eye(self.d) / 2

    def forward(self, A_src, A_tgt, F_src, F_tgt, U_src, U_tgt, w1 = 1, w2 = 1):
        n, m = A_src.shape[1], A_tgt.shape[1]

        # Count number of ones in the adj. matrix to get number of edges
        lambda1 = self.relu(self.lambda1 + self.lambda1.transpose(0, 1)) * w1 
        lambda2 = self.relu(self.lambda2 + self.lambda2.transpose(0, 1)) * w2
        weight = torch.cat((torch.cat((lambda1, lambda2)),
                            torch.cat((lambda2, lambda1))), 1)


        # Init M = [M_1, M_2, .., M_b] with zeros
        M = torch.empty(A_src.shape[0], n*m, n*m, device = F_src.device);

        # Loop over the batch and assemble each M_i
        for i in range(A_src.shape[0]):
            # Get all non-zero entries and build G and H
            thresh = 0.48
            nr_edges1 = torch.sum(A_src[i]>=thresh).to(torch.int32).item() #TODO: Check if putting .item() is okay
            nr_edges2 = torch.sum(A_tgt[i]>=thresh).to(torch.int32).item() #TODO: Check if putting .item() is okay
            G1 = torch.zeros(n, nr_edges1, device = F_src.device)
            H1 = torch.zeros(n, nr_edges1, device = F_src.device)
            G2 = torch.zeros(m, nr_edges2, device = F_src.device)
            H2 = torch.zeros(m, nr_edges2, device = F_src.device)

            entries = (A_src[i] >= 0.49).nonzero()
            for count, (k,j) in enumerate(entries, start=0):
                G1[k, count] = 1
                H1[j, count] = 1
            
            X = reshape_edge_feature_gl(F_src[i], G1, H1)
            entries = (A_tgt[i] >= 0.49).nonzero()
            for count, (k,j) in enumerate(entries, start=0):
                G2[k, count] = 1
                H2[j, count] = 1
            Y = reshape_edge_feature_gl(F_tgt[i], G2, H2)

            # Compute Me and Mp (node-to-node and edge-to-edge similarities)        
            Me = torch.matmul(X.t(), weight)
            Me = torch.matmul(Me, Y)
            Mp = torch.matmul(U_src[i].t(), U_tgt[i])

            # Compute M based on the Affinity Matrix Factorization [Zhou and De Lea Torre]
            K1 = self.kronecker(G2, G1)
            K2 = self.kronecker(H2, H1)

            diagMp = torch.diag(Mp.reshape(-1))
            #diagMe = torch.diag(Me.reshape(-1))
            diagMe = Me.reshape(-1)
            assert K1.shape[1] == diagMe.shape[0]
            K1_new = torch.zeros_like(K1)
            for j in range(K1.shape[1]):
                K1_new[:, j] = K1[:, j] * diagMe[j]
            M[i,:,:] = diagMp + torch.mm(K1_new, K2.t());
            
        return M
    @staticmethod
    def kronecker(A, B):
        return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

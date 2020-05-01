import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
import math

from utils.build_graphs import reshape_edge_feature, kronecker, batch_diagonal
from utils.fgm import kronecker_torch


class Affinity(nn.Module):
    """
    AFFINITY LAYER 
    =============
    
    Arguments:
    ---------
        - A_src, A_tgt: 
            learned graphs from graph learning for input image pair -> (source) and (target)
            --> used to generate G1, G2 and H1 and H2

        - F_src, F_tgt: 
            edge features of input image pair -> (source) and (target)
        
        - U_src, U_tgt:
            node features of input image pair -> (source) and (target)
        
        - d:
            value to initalize weights

        - w1, w2:
            weight scaling factors        

   Outputs:
   -------
        - M: 
            global affinity matrix

    """
    def __init__(self, d):
        """
        AFFINITY LAYER INITIALIZATION
            - Learnable Parameters
            - RELU Unit
        """
        super(Affinity, self).__init__()
        self.d = d
        self.lambda1 = Parameter(Tensor(self.d, self.d))
        self.lambda2 = Parameter(Tensor(self.d, self.d))
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        """
        PARAMETER RESET 
            -  Lambda Parameters
        """
        stdv = 1. / math.sqrt(self.lambda1.size(1) * 2)
        self.lambda1.data.uniform_(-stdv, stdv)
        self.lambda2.data.uniform_(-stdv, stdv)
        self.lambda1.data += torch.eye(self.d) / 2
        self.lambda2.data += torch.eye(self.d) / 2


    def forward(self, A_src, A_tgt,  F_src, F_tgt, U_src, U_tgt, w1 = 1, w2 = 1):
        """ 
        FORWARD ROUTINE
            - Compute global affinity matrix 
        """

        # (A) Retrieve shape parameters
        n, m = F_src.shape[-1], F_tgt.shape[-1]
        batch_num = F_src.shape[0]

        # (B) Construct weight Matrix
        lambda1 = self.relu(self.lambda1 + self.lambda1.transpose(0, 1)) * w1 
        lambda2 = self.relu(self.lambda2 + self.lambda2.transpose(0, 1)) * w2
        weight = torch.cat((torch.cat((lambda1, lambda2)),
                            torch.cat((lambda2, lambda1))), 1)


        # (C) Construct G1, G2 and H1, H2 
        G1 = torch.zeros(batch_num, n, A_src.shape[1], device = F_src.device)#.to(dtype = torch.bool)
        H1 = torch.zeros(batch_num, n, A_src.shape[1], device = F_src.device)#.to(dtype = torch.bool)
        G2 = torch.zeros(batch_num, m, A_tgt.shape[1], device = F_src.device)#.to(dtype = torch.bool)
        H2 = torch.zeros(batch_num, m, A_tgt.shape[1], device = F_src.device)#.to(dtype = torch.bool)

        a = torch.triu_indices(U_src.shape[1], U_src.shape[2], offset = 1)
        b = torch.triu_indices(U_tgt.shape[1], U_tgt.shape[2], offset = 1)
        for i in range(batch_num):
            G1[i, a[0, A_src[i]>0], A_src[i]>0] = 1
            H1[i, a[1, A_src[i]>0], A_src[i]>0] = 1
            G2[i, b[0, A_tgt[i]>0], A_tgt[i]>0] = 1
            H2[i, b[1, A_tgt[i]>0], A_tgt[i]>0] = 1
        

        # (D) Reshape Edge Feature for further use
        X = reshape_edge_feature(F_src, G1, H1)
        Y = reshape_edge_feature(F_tgt, G2, H2)
    
        # (E) Compute Me and Mp (node-to-node and edge-to-edge similarities)   
        M_e = torch.bmm(torch.bmm(X.permute(0,2,1), weight.expand(X.shape[0],-1,-1)), Y)
        M_p = torch.bmm(U_src.permute(0,2,1), U_tgt)

        # (F) Compute M based on the Affinity Matrix Factorization [Zhou and De Lea Torre]
        K1 = kronecker_torch(G2, G1)
        K2 = kronecker_torch(H2, H1)
        
        diagMp = batch_diagonal(M_p.view(M_p.shape[0],-1))
        diagMe = M_e.view(M_e.shape[0],-1, 1)
        K1_new = torch.zeros_like(K1).to(device = K1.device)
        
        for j in range(K1.shape[2]):
            K1_new[:, :, j] = torch.matmul(K1[:, :, j].unsqueeze(2), diagMe[:, j].unsqueeze(2)).squeeze()
        M = diagMp + torch.bmm(K1_new, K2.permute(0,2,1));
            
        return M


    #@staticmethod


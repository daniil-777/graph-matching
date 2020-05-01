import torch.nn as nn
import torch

from GL.affinity import Affinity
from GL.power_iteration import PowerIteration
from utils.sinkhorn import Sinkhorn
from utils.voting_layer import Voting
from GL.displacement_layer import Displacement
from GL.graph_learning import GraphLearning_New
from utils.feature_align import feature_align

from utils.config import cfg

import utils.backbone
CNN = eval('utils.backbone.{}'.format(cfg.BACKBONE))


class Net(CNN):
    """
    GRAPH MATCHING NETWORK
    ======================

    """

    def __init__(self):
        """ 
        NETWORK INIT
        -> inits all layers of the network
        """
        super(Net, self).__init__()
        self.graph_learning = GraphLearning_New(cfg.GMN.FEATURE_CHANNEL, cfg.GMN.NUM_ADJACENCY)
        self.affinity_layer = Affinity(cfg.GMN.FEATURE_CHANNEL)
        self.power_iteration = PowerIteration(max_iter=cfg.GMN.PI_ITER_NUM, stop_thresh=cfg.GMN.PI_STOP_THRESH)
        self.bi_stochastic = Sinkhorn(max_iter=cfg.GMN.BS_ITER_NUM, epsilon=cfg.GMN.BS_EPSILON)
        self.voting_layer = Voting(alpha=cfg.GMN.VOTING_ALPHA)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.GMN.FEATURE_CHANNEL * 2, alpha=cfg.GMN.FEATURE_CHANNEL * 2, beta=0.5, k=0)

    def forward(self, src, tgt, P_src, P_tgt, ns_src, ns_tgt):
        """
        FORWARD ROUTINE
        """

        # (A) Extract node and edge features
        src_node = self.node_layers(src)
        src_edge = self.edge_layers(src_node)
        tgt_node = self.node_layers(tgt)
        tgt_edge = self.edge_layers(tgt_node)

        # (B) Feature Normalization
        src_node = self.l2norm(src_node)
        src_edge = self.l2norm(src_edge)
        tgt_node = self.l2norm(tgt_node)
        tgt_edge = self.l2norm(tgt_edge)

        # (C) FEATURE ARRANGEMENT
        U_src = feature_align(src_node, P_src, ns_src, cfg.PAIR.RESCALE)
        F_src = feature_align(src_edge, P_src, ns_src, cfg.PAIR.RESCALE)
        U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
        F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, cfg.PAIR.RESCALE)


        # (D) Graph Learning on source image
        A_src = self.graph_learning(U_src)

        # (D-1) Optional output debugging of learned graph
        #A_src_ = A_src.clone().detach()
        #A_src_[A_src_==0] = 100
        #print("A_src: \nMin:  ", torch.min(torch.min(A_src_, dim=1).values, dim=1).values.data, ",\nMax:  ", torch.max(torch.max(A_src, dim=1).values, dim=1).values.data, "\nMean: ", torch.mean(A_src, dim=(1,2)))
        
        
        # (E) Graph Learning on target image
        A_tgt = self.graph_learning(U_tgt) #Shared weights, so pass through the same networki
        
        # (E-1) Optional output debugging of learned graph
        #A_tgt_ = A_tgt.clone().detach()
        #A_tgt_[A_tgt_==0] = 100;
        #print("A_tgt: \nMin:  ", torch.min(torch.min(A_tgt_, dim=1).values, dim=1).values.data, ",\nMax:  ", torch.max(torch.max(A_tgt, dim=1).values, dim=1).values.data, "\nMean: ", torch.mean(A_tgt, dim=(1,2)))
        #print("---")


        # (F) Compute Affinity Matrix M
        M = self.affinity_layer(A_src, A_tgt, F_src, F_tgt, U_src, U_tgt)
        #M = self.affinity_layer(G1, H1 G2, H2, F_src, F_tgt, U_src, U_tgt)
        
        # (G) Compute (optimal) assignment vector using power iterations
        v = self.power_iteration(M)
        s = v.view(v.shape[0], P_tgt.shape[1], -1).transpose(1, 2)

        # (H) Apply voting and bi-stochastic layer
        s = self.voting_layer(s, ns_src, ns_tgt)
        s = self.bi_stochastic(s, ns_src, ns_tgt)

        # (I) Compute displacement
        d, _ = self.displacement_layer(s, P_src, P_tgt)
        
        return s, d

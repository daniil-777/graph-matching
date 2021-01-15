import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphLearning(nn.Module):
    """
    GRAPH LEARNING LAYER - ...
    ==============================================

    Arguments:
    ---------

        - n_nodes_len:
            length of node node-feature input 
            --> used for FC initialization

        - node_features:
            Extracted node features

    Outputs:
    -------
        
        - adjacency_matrix:
            Learned Adjacency Matrix
    
    """

    def __init__(self, n_nodes_len):
        """
        INITIALIZATION OF GRAPH LEARNING LAYER
             - Fully-Connected Layer
        """
        super(GraphLearning, self).__init__()
        self.fc1 = nn.Linear(2*n_nodes_len, 1)



    def forward(self, node_features):
        """
        FORWARD ROUTINE
            - computes Adjacency Matrix
        """
        n_nodes = node_features.shape[-1]
        index = torch.arange(n_nodes)
        index_combinations = torch.combinations(index, r=2)
        adjacency_matrix = torch.zeros(node_features.shape[0], n_nodes, n_nodes)
        
        for i,j in index_combinations:
        
            adjacency_matrix[:, j, i] = adjacency_matrix[:, i, j] = torch.sigmoid(self.fc1(torch.cat((node_features[:, :, i], node_features[:, :, j]), dim = 1))).squeeze()
        return adjacency_matrix



class GraphLearning_New(nn.Module):
    """
    GRAPH LEARNING LAYER - ...
    =============================================

    Arguments:
    ---------
        - n_nodes_len:
            length of node-feature input
            --> used for layer initialization

        - max_keypoints:
            maximal number of keypoints

        - node_features:
            Extracted node features

    Outputs:
    -------

    """
    def __init__(self, n_nodes_len, max_keypoints):
        """
        INITIALIZATION OF GRAPH LEARNING LAYER
        """
        super(GraphLearning_New, self).__init__()
        self.layer1 = nn.Conv1d(n_nodes_len, max_keypoints, 1)



    def forward(self, node_features):
        """
        FORWARD ROUTINE
        """
        result = F.relu(self.layer1(node_features))
        result = result[:, :result.shape[2], :]
        upper_matrix = torch.triu(torch.ones(1, result.shape[2], result.shape[2]), diagonal = 1).to(device=node_features.device, dtype = torch.bool)

        adjacency_matrix = torch.masked_select(result, upper_matrix).view(result.shape[0], -1)

        return adjacency_matrix




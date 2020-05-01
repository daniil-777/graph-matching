import torch
import torch.nn as nn


class Displacement(nn.Module):
    """
    DISPLACEMENT LAYER
    ==================

    Arguments:
    ---------
        - s:
            double stochastic matrix s

        - P_src:
            point-set on source image

        - P_tgt:
            point-set on target image

        - ns_gt:
            ground truth points in source image

    Outputs:
    --------

        - d:
            displacement vector

        - grad_mask:
            mask for dummy nodes (not calculate if ns_gt=None)

    """
    def __init__(self):
        """
        DISPLACEMENT LAYER INITIALIZATION
        """
        super(Displacement, self).__init__()

    def forward(self, s, P_src, P_tgt, ns_gt=None):
        """
        FORWARD ROUTINE
        """
        if ns_gt is None:
            max_n = s.shape[1]
            P_src = P_src[:, 0:max_n, :]
            grad_mask = None
        else:
            grad_mask = torch.zeros_like(P_src)
            for b, n in enumerate(ns_gt):
                grad_mask[b, 0:n] = 1

        d = torch.matmul(s, P_tgt) - P_src
        return d, grad_mask

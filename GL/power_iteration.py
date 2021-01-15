import torch
import torch.nn as nn
from utils.sparse import sbmm


class PowerIteration(nn.Module):
    """
    POWER ITERATION LAYER
    =====================

    Arguments:
    ---------
        - max_iter:
            maximal numbers of iteration

        - stop_thresh:
            early stopping threshold

        - M:
            affinity matrix

    Outputs:
    -------
        - v*:
            optimal assignment vector

    """

    def __init__(self, max_iter=50, stop_thresh=2e-7):
        """
        INITIALIZATION OF LAYER
        """
        super(PowerIteration, self).__init__()
        self.max_iter = max_iter
        self.stop_thresh = stop_thresh

    def forward(self, M, v0=None):
        """
        FORWARD ROUTINE
        """
        
        # (A) Retrieve shape parameters
        batch_num = M.shape[0]
        mn = M.shape[1]

        # (B) Init assignment vector
        if v0 is None:
            v0 = torch.ones(batch_num, mn, 1, dtype=M.dtype, device=M.device)

        # (C) Perform max-iter power iterations
        v = vlast = v0
        for i in range(self.max_iter):
            if M.is_sparse:
                v = sbmm(M, v)
            else:
                v = torch.bmm(M, v)
            n = torch.norm(v, p=2, dim=1)
            v = torch.matmul(v, (1 / n).view(batch_num, 1, 1))
            if torch.norm(v - vlast) < self.stop_thresh:
                return v.view(batch_num, -1)
            vlast = v

        return v.view(batch_num, -1)


if __name__ == '__main__':
    from torch.autograd import gradcheck
    input = (torch.randn(3, 40, 40, dtype=torch.double, requires_grad=True),)

    pi = PowerIteration()

    test = gradcheck(pi, input, eps=1e-6, atol=1e-4)
    print(test)

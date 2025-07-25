import torch
from torch import nn
import torch.nn.functional as F

from torch_scatter import segment_csr
from mlp import MLPLinear

class IntegralTransform(nn.Module):
    """Integral Kernel Transform (GNO)
    Computes one of the following:
        (a) \int_{A(x)} k(x, y) dy
        (b) \int_{A(x)} k(x, y) * f(y) dy
        (c) \int_{A(x)} k(x, y, f(y)) dy
        (d) \int_{A(x)} k(x, y, f(y)) * f(y) dy

    x : Points for which the output is defined
    y : Points for which the input is defined
    A(x) : A subset of all points y (depending on 
           each x) over which to integrate
    k : A kernel parametrized as a MLP
    f : Input function to integrate against given 
        on the points y

    If f is not given, a transform of type (a)
    is computed. Otherwise transforms (b), (c),
    or (d) are computed. The sets A(x) are specified
    as a graph in CRS format.

    Parameters
    ----------
    mlp : torch.nn.Module, default None
        MLP parametrizing the kernel k. Input dimension
        should be dim x + dim y or dim x + dim y + dim f
    mlp_layers : list, default None
        List of layers sizes speficing a MLP which 
        parametrizes the kernel k. The MLP will be 
        instansiated by the MLPLinear class  
    mlp_non_linearity : callable, default torch.nn.functional.gelu
        Non-linear function used to be used by the 
        MLPLinear class. Only used if mlp_layers is 
        given and mlp is None
    transform_type : int, default 0
        Which integral transform to compute. The mapping is:
        0 -> (b), 1 -> (c), 2 -> (d). If the input f is not
        given then (a) is computed by default independently 
        of this parameter.
    """
    def __init__(self, 
                 mlp=None,
                 mlp_layers=None,
                 mlp_non_linearity=F.gelu, 
                 transform_type=0
                 ):
        
        super().__init__()

        assert mlp is not None or mlp_layers is not None
        assert transform_type >= 0 and transform_type <= 2

        self.transform_type = transform_type

        if mlp is None:
            self.mlp = MLPLinear(layers=mlp_layers,
                                 non_linearity=mlp_non_linearity)
        else:
            self.mlp = mlp
        
    """"
    

    Assumes x=y if not specified
    Integral is taken w.r.t. the neighbors
    If no weights are given, a Monte-Carlo approximation is made
    NOTE: For transforms of type 0 or 2, out channels must be
    the same as the channels of f
    """
    def forward(self, 
                y, 
                neighbors,
                x=None, 
                f_y=None, 
                weights=None
            ):
        """Compute a kernel integral transform

        Parameters
        ----------
        y : torch.Tensor of size [n, d1]
            n points of dimension d1 specifying 
            the space to integrate over.
        neighbors : dict
            The sets A(x) given in CRS format. The 
            dict must contain the keys "neighbors_index"
            and "neighbors_row_splits." For descriptions
            of the two, see NeighborSearch.
        x : torch.Tensor of size [m, d2], default None
            m points of dimension d2 over which the 
            output function is defined. If None,
            x = y.
        f_y : torch.Tensor of size [n, d3], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output size must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        weights : torch.Tensor of size [n,], default None
            Weights for each point y proprtional to the 
            volume around f(y) being integrated. For example,
            suppose d1=1 and let y_1 < y_2 < ... < y_{n+1}
            be some points. Then, for a Riemann sum,
            the weights are y_{j+1} - y_j. If None,
            1/|A(x)| is used.
        
        Output
        ----------
        out_features : torch.Tensor of size [m, d4]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """
        
        if x is None:
            x = y

        rep_features = y[neighbors['neighbors_index']]
        if f_y is not None:
            in_features = f_y[neighbors['neighbors_index']]

        num_reps = neighbors['neighbors_row_splits'][1:] - neighbors['neighbors_row_splits'][:-1]
        self_features = torch.repeat_interleave(x, num_reps, dim=0)

        agg_features = torch.cat([rep_features, self_features], dim=1)
        if f_y is not None and (self.transform_type == 1 or self.transform_type == 2):
            agg_features = torch.cat([agg_features, in_features], dim=1)

        rep_features = self.mlp(agg_features)

        if f_y is not None and self.transform_type != 1:
            rep_features = rep_features*in_features 

        if weights is not None:
            rep_features = weights[neighbors['neighbors_index']]*rep_features
            reduction = 'sum'
        else:
            reduction = 'mean'

        out_features = segment_csr(rep_features, 
                                   neighbors['neighbors_row_splits'], 
                                   reduce=reduction)
        return out_features
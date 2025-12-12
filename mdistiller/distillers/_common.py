import torch
import torch.nn as nn
import torch.nn.functional as F


def get_feat_shapes(student, teacher, input_size):
    data = torch.randn(1, 3, *input_size)
    with torch.no_grad():
        feat_s = None if student is None else student(data)[1]
        feat_t = None if teacher is None else teacher(data)[1]
    feat_s_shapes = None if feat_s is None else [f.shape for f in feat_s["feats"]]
    feat_t_shapes = None if feat_t is None else [f.shape for f in feat_t["feats"]]
    return feat_s_shapes, feat_t_shapes


def compute_mapped_layers(layers, teacher, student, verbose: bool=False):
    t_layers = len(teacher.blocks)
    s_layers = len(student.blocks)

    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        
        t, s = np.meshgrid(
            np.divide(layers, t_layers),
            np.linspace(0, 1, s_layers),
            indexing='ij',
        )
        dist = np.abs(t - s)
        m_layers_stu = linear_sum_assignment(dist)[1].tolist()
    except ImportError:
        map_fact = t_layers // s_layers
        m_layers_stu = [l // map_fact for l in layers] if map_fact != 1 else layers
        
    if verbose:
        print(f"Distill Teacher {layers} to Student {m_layers_stu}")
    return m_layers_stu


class ConvReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


class SimpleAdapter(nn.Module):
    def __init__(self, s_features, t_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        in_features = s_features
        out_features = t_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        return x


class SiNGERAdapter(nn.Module):
    """
    Null-space LoRA Adapter.
      • method = "singer" : Use the null-space for init params.
      • method = "random" : Use a randomly generated params for init params.
    """
    def __init__(self, W: torch.Tensor, rank: int = 16, threshold: float = 1.0E-3, method: str = "singer"):
        super().__init__()
        self.threshold = threshold
        d = W.size(0)
        method = method.lower()
        U, S, _ = torch.linalg.svd(W, full_matrices=True)
        small_indices = torch.nonzero(S < threshold, as_tuple=False).squeeze(-1)
        N_raw = U[:, small_indices].t()
        k = N_raw.size(0)
        if method == "singer":
            if k == 0:
                import warnings
                warnings.warn("No explicit null-space detected; using random basis.")
                N = F.normalize(torch.randn(rank, d, device=W.device, dtype=W.dtype), dim=-1)
            else:
                N = N_raw
                if k < rank:
                    pad = F.normalize(torch.randn(rank - k, d, device=W.device, dtype=W.dtype), dim=-1)
                    N = torch.cat([N, pad], dim=0)
                else:
                    N = N[:rank]
        elif method == "random":
            N = F.normalize(torch.randn(rank, d, device=W.device, dtype=W.dtype), dim=-1)
        else:
            raise ValueError(f'Unsupported method "{method}". Use "singer" or "random".')

        self.W_d = nn.Parameter(N.t().clone())   # [d, r]
        self.W_u = nn.Parameter(N.clone())       # [r, d]

    def forward(self, A: torch.Tensor, return_delta: bool=False) -> torch.Tensor:
        delta = (A @ self.W_d) @ self.W_u
        if return_delta:
            return A + delta, delta
        else:
            return A + delta



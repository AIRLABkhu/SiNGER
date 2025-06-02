import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from ._base import Distiller
from ._common import SimpleAdapter, get_feat_shapes


class Generator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
    
    @property
    def dim(self):
        return self.conv1.in_channels
    
    def forward(self, x: torch.Tensor):
        num_tokens = x.size(1)
        num_rows = int(num_tokens**0.5)
        
        cls_token, spa_tokens = torch.split(x, [1, num_tokens-1], dim=1)
        spa_tokens = einops.rearrange(spa_tokens, 'b (h w) d -> b d h w', h=num_rows)
        gen_tokens = self.conv2(self.relu(self.conv1(spa_tokens)))
        
        return torch.cat([
            cls_token, einops.rearrange(
                gen_tokens, 'b d h w -> b (h w) d'
            )
        ], dim=1)


class ViTKD(Distiller):
    """ViTKD: Feature-based Knowledge Distillation for Vision Transformers"""

    def __init__(self, student, teacher, cfg):
        super(ViTKD, self).__init__(student, teacher)
        self.masking_ratio = cfg.VITKD.MASKING_RATIO
        if cfg.VITKD.REF_AMD:
            self.m_layers = cfg.AMD.M_LAYERS + [len(self.teacher.get_layers()) - 1]
            self.layers = []
            self.loss_weight = cfg.AMD.LOSS.FEAT_WEIGHT
        else:
            self.m_layers = cfg.VITKD.M_LAYERS + [len(self.teacher.get_layers()) - 1]
            self.layers = cfg.VITKD.LAYERS
            self.loss_weight = 1.0
        assert len(set(self.m_layers).intersection(self.layers)) == 0
        
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.VITKD.INPUT_SIZE
        )
        self.adapters = nn.ModuleDict({
            f'{lidx:02d}': SimpleAdapter(feat_s_shapes[lidx][-1], feat_t_shapes[lidx][-1])
            for lidx in self.layers + self.m_layers
        })
        self.generators = nn.ModuleDict({
            f'{lidx:02d}': Generator(feat_t_shapes[lidx][-1])
            for lidx in self.m_layers
        })

    def get_learnable_parameters(self):
        yield from super().get_learnable_parameters()
        yield from self.adapters.parameters()

    def get_extra_parameters(self):
        return sum(map(torch.Tensor.numel, self.adapters.parameters()))

    def forward_train(self, image, target, **kwargs):
        _, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)
        
        loss: torch.Tensor = 0.0
        for layer in self.layers:
            adapter = self.adapters[f'{layer:02d}']
            loss = loss + F.mse_loss(
                adapter(feature_student['feats'][layer]),
                feature_teacher['feats'][layer],
            )
        
        for layer in self.m_layers:
            adapter = self.adapters[f'{layer:02d}']
            generator = self.generators[f'{layer:02d}']
            a_s = adapter(feature_student['feats'][layer])
            g_s = generator(a_s)
            mask = torch.rand_like(g_s[..., 0:1], device=g_s.device) <= self.masking_ratio
            mask = mask.expand_as(g_s)
            loss = loss + F.mse_loss(
                g_s[mask], feature_teacher['feats'][layer][mask],
            )
        
        loss = loss * self.loss_weight

        # losses
        losses_dict = {
            "loss_kd": loss,
        }
        return torch.zeros(image.size(0), 1000, dtype=loss.dtype, device=loss.device), losses_dict

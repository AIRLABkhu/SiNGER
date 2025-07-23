import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import {
    get_feat_shapes,
    SimpleAdapter,
    make_zscore_mask,
    make_gaussian_std_mask,
    compute_mapped_layers}

class AMD_MASK(Distiller):
    """Artifact Manipulating Distillation"""
    def __init__(self, student, teacher, cfg):
        super(AMD_MASK, self).__init__(student, teacher)
        self.feat_loss_weight = cfg.AMD.LOSS.FEAT_WEIGHT
        self.m_layers = cfg.AMD.M_LAYERS + [len(self.teacher.get_layers()) - 1]
        self.m_layers_stu = compute_mapped_layers(self.m_layers, self.teacher, self.student)
        
        self.align_type = cfg.AMD.ALIGN_TYPE
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.AMD.INPUT_SIZE
        )
        # Adapters from Student to Teacher
        self.adapter_dict = nn.ModuleDict({
            **{
                f"adapter_{m_l_stu:03d}": SimpleAdapter(feat_s_shapes[m_l_stu][-1], feat_t_shapes[m_l][-1])
                for m_l_stu, m_l in zip(self.m_layers_stu, self.m_layers)
            }
        })

        self.af_enabled = cfg.AMD.AF.ENABLE
        self.af_type = cfg.AMD.AF.CRITERIA.TYPE
        self.af_threshold = cfg.AMD.AF.CRITERIA.THRES

    def get_learnable_parameters(self):
        yield from super().get_learnable_parameters()
        yield from self.adapter_dict.parameters()

    def get_extra_parameters(self):
        return sum(
            sum(map(torch.Tensor.numel, module.parameters()))
            for module in [
                self.adapter_dict,
            ]
        )

    def forward_train(self, image, target, **kwargs):
        _, feature_student = self.student.forward_wohead(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher.forward_wohead(image)
            
        loss_feat = 0.0
        for m_l_stu, m_l in zip(self.m_layers_stu, self.m_layers):
            f_s = feature_student["feats"][m_l_stu]
            f_t = feature_teacher["feats"][m_l]
            if self.af_enabled:
                match self.af_type:
                    case 'zscore':
                        outlier_mask = make_zscore_mask(f_t, threshold=self.af_threshold)
                    case 'gaussian_std':
                        outlier_mask = make_gaussian_std_mask(f_t)
                    case 'quantile':
                        cls, patch = f_t[:, :1], f_t[:, 1:]
                        norms = patch.norm(dim=-1).detach()
                        q_threshold = torch.quantile(norms, 0.95, dim=1, keepdim=True)
                        patch_mask = (norms > q_threshold)
                        cls_mask = torch.zeros((patch_mask.size(0), 1),                 # (B, 1)
                                                dtype=patch_mask.dtype,
                                                device=patch_mask.device)
                        outlier_mask = torch.cat((cls_mask, patch_mask), dim=1)
                    case _:
                        raise NotImplementedError(self.af_type)
                inlier_bool_mask = outlier_mask.bool().logical_not()
                f_s = self.adapter_dict[f"adapter_{m_l_stu:03d}"](f_s)
                f_s_inliers = f_s[inlier_bool_mask]
                f_t_inliers = f_t[inlier_bool_mask]
                loss_feat_mse = F.mse_loss(f_s_inliers, f_t_inliers)
                loss_feat = loss_feat + loss_feat_mse
            else:
                f_s = self.adapter_dict[f"adapter_{m_l_stu:03d}"](f_s)
                loss_feat = loss_feat + F.mse_loss(f_s, f_t)
        loss_feat = self.feat_loss_weight * loss_feat / len(self.m_layers) 

        losses_dict = {
            "loss_kd": loss_feat,
        }

        return torch.zeros(f_s.size(0), 1000, dtype=f_s.dtype, device=f_s.device), losses_dict



import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import get_feat_shapes, SimpleAdapter, compute_mapped_layers


class FitViT(Distiller):
    """from FitNets: Hints for Thin Deep Nets"""

    def __init__(self, student, teacher, cfg):
        super(FitViT, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.FITNET.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.FITNET.LOSS.FEAT_WEIGHT
        self.hint_layer = [*cfg.FITVIT.M_LAYERS, len(self.teacher.get_layers()) - 1]
        self.hint_layer_stu = compute_mapped_layers(self.hint_layer, self.teacher, self.student)
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.DATASET.INPUT_SIZE
        )
        self.adapters = nn.ModuleDict({
            f"adapter_{m_l_stu:03d}": SimpleAdapter(feat_s_shapes[m_l_stu][-1], feat_t_shapes[m_l][-1])
            for m_l_stu, m_l in zip(self.hint_layer_stu, self.hint_layer)
        })

        self.af_enabled = cfg.FITNET.AF.ENABLE
        self.af_type = cfg.FITNET.AF.CRITERIA.TYPE
        self.af_threshold = cfg.FITNET.AF.CRITERIA.THRES

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.adapters.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.adapters.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher.forward_wohead(image)
            
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_feat: torch.Tensor = 0.0
        for m_l_stu, m_l in zip(self.hint_layer_stu, self.hint_layer):
            adapter: SimpleAdapter = self.adapters[f"adapter_{m_l_stu:03d}"]
            f_s = feature_student['feats'][m_l_stu]
            f_t = feature_teacher['feats'][m_l]
            loss_feat = loss_feat + F.mse_loss(adapter.forward(f_s), f_t)
        loss_feat = self.feat_loss_weight * loss_feat

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
        }
        return logits_student, losses_dict

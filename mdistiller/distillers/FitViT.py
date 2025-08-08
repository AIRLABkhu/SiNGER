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
        self.hint_layer = cfg.FITNET.HINT_LAYER
        self.hint_layer_stu = compute_mapped_layers([self.hint_layer], self.teacher, self.student)[0]
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )
        self.adapter = SimpleAdapter(
            feat_s_shapes[self.hint_layer_stu][-1], feat_t_shapes[self.hint_layer][-1]
        )

        self.af_enabled = cfg.FITNET.AF.ENABLE
        self.af_type = cfg.FITNET.AF.CRITERIA.TYPE
        self.af_threshold = cfg.FITNET.AF.CRITERIA.THRES

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.adapter.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.adapter.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            feature_teacher = self.teacher.forward_partial(image, self.hint_layer)
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        f_s = self.adapter(feature_student["feats"][self.hint_layer_stu])
        loss_feat = self.feat_loss_weight * F.mse_loss(
            f_s, feature_teacher["feats"][self.hint_layer]
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
        }
        return logits_student, losses_dict

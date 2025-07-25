import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import (
    get_feat_shapes, 
    SimpleAdapter, 
    SNERAdapter,
    compute_mapped_layers
)

def init_sner(model, layers, rank, sner_method):
    return nn.ModuleDict({
        **{
                f"sner_{l:03d}": SNERAdapter((model.blocks[l].mlp.fc2.weight @ model.blocks[l].mlp.fc1.weight).t().detach(),
                                             rank=rank, method=sner_method)
                for l in layers
            }
    })
    
class AMD_SNER(Distiller):
    def __init__(self, student, teacher, cfg):
        super(AMD_SNER, self).__init__(student, teacher)
        self.feat_loss_weight = cfg.AMD.LOSS.FEAT_WEIGHT
        self.outlier_loss_weight = cfg.AMD.LOSS.OUTLIER_WEIGHT
        self.info_loss_weight = cfg.AMD.LOSS.INFO_WEIGHT
        
        self.rank = cfg.AMD.SNER.RANK
        self.outlier_q = cfg.AMD.SNER.OUTLIER_Q
        self.sner_method = cfg.AMD.SNER.METHOD
        self.m_layers = cfg.AMD.M_LAYERS + [len(self.teacher.get_layers()) - 1]
        self.m_layers_stu = compute_mapped_layers(self.m_layers, self.teacher, self.student)
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
        
        # SNER LoRA for Teacher
        self.sner_dict = init_sner(self.teacher, self.m_layers, self.rank, self.sner_method)
        
    def get_learnable_parameters(self):
        yield from super().get_learnable_parameters()
        yield from self.adapter_dict.parameters()
        yield from self.sner_dict.parameters()
        
    def get_extra_parameters(self):
        return sum(
            sum(map(torch.Tensor.numel, module.parameters()))
            for module in [
                self.adapter_dict,
                self.sner_dict,
            ]
        )
        
    def forward_train(self, image, target, **kwargs):
        _, feature_student = self.student.forward_wohead(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher.forward_wohead(image)
            
        loss_feat, loss_outlier, loss_info = 0.0, 0.0, 0.0    
        for m_l_stu, m_l in zip(self.m_layers_stu, self.m_layers):
            
            f_s = feature_student["feats"][m_l_stu ]# F_S^l
            f_t = feature_teacher["feats"][m_l]  # F_T^l
            
            cls, patch = f_t[:, :1], f_t[:, 1:]
            patch_sner = self.sner_dict[f"sner_{m_l:03d}"](patch)
            f_t_sner = torch.cat((cls, patch_sner), 1)  # \hat{F}_T^l

            # 3.1. Knowledge Distillation Loss (L_KD)
            proj_f_s = self.adapter_dict[f"adapter_{m_l_stu:03d}"](f_s)
            loss_feat = loss_feat + F.mse_loss(proj_f_s, f_t_sner) # \hat{F}_T^l - F_S^l

            # 3.2. Outlier Suppression Loss (L_outlier)
            norms = f_t_sner[:, 1:].norm(dim=-1)
            q_threshold = torch.quantile(norms, self.outlier_q, dim=1, keepdim=True).detach()
            outlier_norms = norms[norms > q_threshold]
            
            if outlier_norms.numel() > 0:
                target_norms = q_threshold.expand_as(norms)[norms > q_threshold]
                loss_outlier = loss_outlier + F.mse_loss(outlier_norms, target_norms)
            else:
                loss_outlier = loss_outlier + torch.tensor(0.0, device=f_s.device)

            # 3.3. Information Preservation Loss (L_info)
            with torch.no_grad():
                f_t_next = self.teacher.blocks[m_l + 1].forward(f_t) # F_T^{l+1}
            
            f_t_sner_next = self.teacher.blocks[m_l + 1].forward(f_t_sner) # \hat{F}_T^{l+1}
            cosine_sim = F.cosine_similarity(f_t_sner_next, f_t_next, dim=-1) # \hat{F}_T^{l+1} - F_T^{l+1}
            loss_info = loss_info + (1.0 - cosine_sim).mean()
            
        loss_feat = self.feat_loss_weight * loss_feat / len(self.m_layers) 
        loss_info = self.info_loss_weight * loss_info / len(self.m_layers) 
        loss_outlier = self.outlier_loss_weight * loss_outlier / len(self.m_layers) 
   
        losses_dict = {
            "loss_kd": loss_feat,
            "loss_info": loss_info,
            "loss_outlier": loss_outlier,
        }
        
        return torch.zeros(f_s.size(0), 1000, dtype=f_s.dtype, device=f_s.device), losses_dict
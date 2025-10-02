from .AMD_SNER import (
    torch,
    F,
    SNERAdapter,
    AMD_SNER,
)

class AMD_SNER_v2(AMD_SNER):
    def __init__(self, student, teacher, cfg):
        super(AMD_SNER_v2, self).__init__(student, teacher, cfg)
        self.ortho_loss_weight = cfg.AMD.LOSS.ORTHO_WEIGHT
        self.nullspace_loss_weight = cfg.AMD.LOSS.NULLSPACE_WEIGHT
    
    def forward_train(self, image, target, **kwargs):
        _, feature_student = self.student.forward_wohead(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher.forward_wohead(image)
            
        loss_feat, loss_outlier, loss_info = 0.0, 0.0, 0.0
        loss_ortho, loss_nullspace = 0.0, 0.0
        for m_l_stu, m_l in zip(self.m_layers_stu, self.m_layers):
            
            f_s = feature_student["feats"][m_l_stu]# F_S^l
            f_t = feature_teacher["feats"][m_l]  # F_T^l
            
            cls, patch = f_t[:, :1], f_t[:, 1:]
            adapter: SNERAdapter = self.sner_dict[f"sner_{m_l:03d}"]
            patch_sner, delta = adapter.forward(patch, return_delta=True)
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
            if m_l == self.m_layers[-1]: # if last layer
                cosine_sim = F.cosine_similarity(f_t_sner, f_t, dim=-1)  # \hat{F}_T^{l} - F_T^{l}   
            else:
                with torch.no_grad(): ## else itermediate_layer:
                    f_t_next = self.teacher.blocks[m_l + 1].forward(f_t)  # F_T^{l+1}
                f_t_sner_next = self.teacher.blocks[m_l + 1].forward(f_t_sner)  # \hat{F}_T^{l+1}
                cosine_sim = F.cosine_similarity(f_t_sner_next, f_t_next, dim=-1)  # \hat{F}_T^{l+1} - F_T^{l+1}
            
            loss_info = loss_info + (1.0 - cosine_sim).mean()
            
            # 3.x1. Orthogonality loss
            sner_weight = adapter.W_d @ adapter.W_u
            wwT = sner_weight @ sner_weight.T
            eye = torch.eye(*wwT.shape, dtype=wwT.dtype, device=wwT.device)
            loss_ortho = loss_ortho + F.mse_loss(wwT, eye)
            
            # 3.x2. Null space loss
            if m_l < len(self.teacher.blocks) - 1:
                delta_next = self.teacher.blocks[m_l + 1].forward(delta)
                zero = torch.zeros_like(delta_next)
                loss_nullspace = loss_nullspace + F.mse_loss(delta_next, zero)
            else:
                delta_next = torch.tensor(0.0, dtype=image.dtype, device=image.device)
        
        n_layers = len(self.m_layers)
        loss_feat = self.feat_loss_weight * loss_feat / n_layers
        loss_info = self.info_loss_weight * loss_info / n_layers
        loss_outlier = self.outlier_loss_weight * loss_outlier / n_layers
        loss_ortho = self.ortho_loss_weight * loss_ortho / n_layers
        loss_nullspace = self.nullspace_loss_weight * loss_nullspace / (n_layers - 1)
        
        losses_dict = {
            "loss_kd": loss_feat,
            "loss_info": loss_info,
            "loss_outlier": loss_outlier,
            "loss_ortho": loss_ortho,
            "loss_nullspace": loss_nullspace,
        }
        
        return torch.zeros(f_s.size(0), 1000, dtype=f_s.dtype, device=f_s.device), losses_dict

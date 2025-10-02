from .AMD_SNER import (
    torch,
    F,
    AMD_SNER,
)

class AMD_SNER_v3(AMD_SNER):
    def forward_train(self, image, target, **kwargs):
        _, feature_student = self.student.forward_wohead(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher.forward_wohead(image)
        
        loss_feat    = 0.0
        loss_outlier = 0.0
        loss_info    = 0.0

        inter_feat = None

        for m_l_stu, m_l in zip(self.m_layers_stu, self.m_layers):
            
            f_s = feature_student["feats"][m_l_stu]# F_S^l
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
                layer_mse_loss = F.mse_loss(outlier_norms, target_norms)
                loss_outlier = loss_outlier + layer_mse_loss
            else:
                loss_outlier = loss_outlier + torch.tensor(0.0, device=f_s.device)    
            

            if m_l != self.m_layers[-1]:
                with torch.no_grad(): ## else itermediate_layer:
                    f_t_next = self.teacher.blocks[m_l + 1].forward(f_t)  # F_T^{l+1}
                f_t_sner_next = self.teacher.blocks[m_l + 1].forward(f_t_sner)  # \hat{F}_T^{l+1}

                inter_feat = f_t_sner
                loss_info = loss_info + F.mse_loss(f_t_next, f_t_sner_next)
            else: 
                patches_hat = f_t_sner[:, 1:, :] 
                patches_inter = inter_feat[:,      1:, :]  # inter or original
                
                A = F.normalize(patches_hat, dim=-1)           # [B,P,D]
                B = F.normalize(patches_inter, dim=-1)

                G_hat = A @ A.transpose(-1, -2)                # [B,P,P]
                G_int = B @ B.transpose(-1, -2)

                loss_info = loss_info + F.mse_loss(G_hat, G_int)

        
        loss_feat = self.feat_loss_weight * loss_feat / len(self.m_layers) 
        loss_info = self.info_loss_weight * loss_info / len(self.m_layers) 
        loss_outlier = self.outlier_loss_weight * loss_outlier / len(self.m_layers) 

        losses_dict = {
            "loss_kd": loss_feat,
            "loss_info": loss_info,
            "loss_outlier": loss_outlier,
        }

        return torch.zeros(f_s.size(0), 1000, dtype=f_s.dtype, device=f_s.device), losses_dict

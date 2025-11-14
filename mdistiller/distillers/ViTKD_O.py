import torch
import torch.nn as nn

try:
    from ._base import Distiller
    from ._common import get_feat_shapes, compute_mapped_layers
except:
    from mdistiller.distillers._base import Distiller
    from mdistiller.distillers._common import get_feat_shapes, compute_mapped_layers


class ViTKDLoss(nn.Module):

    """PyTorch version of `ViTKD: Practical Guidelines for ViT feature knowledge distillation` """

    def __init__(self,
                 student_dims,
                 teacher_dims,
                 alpha_vitkd=0.00003,
                 beta_vitkd=0.000003,
                 lambda_vitkd=0.5,
                 ):
        super(ViTKDLoss, self).__init__()
        self.alpha_vitkd = alpha_vitkd
        self.beta_vitkd = beta_vitkd
        self.lambda_vitkd = lambda_vitkd
    
        if student_dims != teacher_dims:
            self.align2 = nn.ModuleList([
                nn.Linear(student_dims, teacher_dims, bias=True)
                for i in range(2)])
            self.align = nn.Linear(student_dims, teacher_dims, bias=True)
        else:
            self.align2 = None
            self.align = None

        self.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_dims))

        self.generation = nn.Sequential(
                nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), 
                nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1))

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(List): [B*2*N*D, B*N*D], student's feature map
            preds_T(List): [B*2*N*D, B*N*D], teacher's feature map
        """
        low_s = preds_S[0]
        low_t = preds_T[0]
        high_s = preds_S[1]
        high_t = preds_T[1]
        
        if low_s.size(-2) != low_t.size(-2):  # optionally skip prefix tokens
            n_std = low_s.size(-2)
            cls_token = low_t[..., :1, :]
            patch_tokens = low_t[..., -n_std+1:, :]
            low_t = torch.cat([cls_token, patch_tokens], dim=-2)
        
        if high_s.size(-2) != high_t.size(-2):  # optionally skip prefix tokens
            n_std = high_s.size(-2)
            cls_token = high_t[..., :1, :]
            patch_tokens = high_t[..., -n_std+1:, :]
            high_t = torch.cat([cls_token, patch_tokens], dim=-2)

        B = low_s.shape[0]
        loss_mse = nn.MSELoss(reduction='sum')

        '''ViTKD: Mimicking'''
        if self.align2 is not None:
            for i in range(2):
                if i == 0:
                    xc = self.align2[i](low_s[:,i]).unsqueeze(1)
                else:
                    xc = torch.cat((xc, self.align2[i](low_s[:,i]).unsqueeze(1)),dim=1)
        else:
            xc = low_s

        loss_lr = loss_mse(xc, low_t) / B * self.alpha_vitkd

        '''ViTKD: Generation'''
        if self.align is not None:
            x = self.align(high_s)
        else:
            x = high_s

        # Mask tokens
        B, N, D = x.shape
        x, mat, ids, ids_masked = self.random_masking(x, self.lambda_vitkd)
        mask_tokens = self.mask_token.repeat(B, N - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, D))
        mask = mat.unsqueeze(-1)

        hw = int(N**0.5)
        x = x.reshape(B, hw, hw, D).permute(0, 3, 1, 2)
        x = self.generation(x).flatten(2).transpose(1,2)

        loss_gen = loss_mse(torch.mul(x, mask), torch.mul(high_t, mask))
        loss_gen = loss_gen / B * self.beta_vitkd / self.lambda_vitkd
            
        return loss_lr + loss_gen

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_masked = ids_shuffle[:, len_keep:L]

        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore, ids_masked


class ViTKD_O(Distiller):
    """ViTKD: Feature-based Knowledge Distillation for Vision Transformers"""

    def __init__(self, student, teacher, cfg):
        super(ViTKD_O, self).__init__(student, teacher)
        self.loss_weight = 1.0

        # feature shapes
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.DATASET.INPUT_SIZE
        )

        # 레이어 매핑
        self.layers = [0, 2]
        self.m_layers = [len(self.teacher.get_layers()) - 1]
        self.layers_stu = compute_mapped_layers(self.layers, self.teacher, self.student)
        self.m_layers_stu = compute_mapped_layers(self.m_layers, self.teacher, self.student)

        with open('_temp/temp.yaml', 'w') as file:
            print(cfg, file=file)
        # Loss 모듈 초기화 (마지막 레이어 차원 기준)
        self.vitkd_loss = ViTKDLoss(
            student_dims=feat_s_shapes[-1][-1],
            teacher_dims=feat_t_shapes[-1][-1],
            alpha_vitkd=cfg.VITKD.HPARAMS.ALPHA,
            beta_vitkd=cfg.VITKD.HPARAMS.BETA,
            lambda_vitkd=cfg.VITKD.MASKING_RATIO,
        )

    def forward_train(self, image, target, **kwargs):
        _, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)

        # 여러 개 레이어를 cfg 기반으로 가져오기
        preds_S, preds_T = [
            torch.stack([feature_student['feats'][lidx][..., 1:, :] for lidx in self.layers_stu], dim=1),
            feature_student['feats'][self.m_layers_stu[0]][..., 1:, :],
        ], [
            torch.stack([feature_teacher['feats'][lidx][..., 1:, :] for lidx in self.layers], dim=1),
            feature_teacher['feats'][self.m_layers[0]][..., 1:, :],
        ]

        loss = self.vitkd_loss(preds_S, preds_T) * self.loss_weight

        return torch.zeros(image.size(0), 1000, dtype=loss.dtype, device=loss.device), {
            "loss_kd": loss,
        }


if __name__ == '__main__':
    from mdistiller.models.imagenet.vit import vit_large_patch16_224, vit_tiny_patch16_224
    from mdistiller.distillers.ViTKD_O import ViTKD_O
    from mdistiller.engine.cfg import CFG as cfg
    import torch

    cfg.VITKD.LAYERS = [0, 2]
    student = vit_tiny_patch16_224()
    teacher = vit_large_patch16_224()
    distiller = ViTKD_O(student, teacher, cfg)

    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(low=0, high=1000, size=(2,))
    _, loss = distiller.forward_train(x, y)
    print(loss['loss_kd'])

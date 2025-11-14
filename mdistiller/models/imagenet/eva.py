import os
from typing import Literal, Callable
from functools import partial
import torch
from torch import nn
from timm.models.eva import (
    register_model,
    build_model_with_cfg,
    checkpoint_filter_fn,
    Eva as TimmEva,
    LayerNorm,
)
from .._base import ModelBase


class Eva(TimmEva, ModelBase):
    def __init__(
            self,
            img_size: int | tuple[int, int] = 224,
            patch_size: int | tuple[int, int] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            swiglu_align_to: int = 0,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            attn_type: str = 'eva',
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = LayerNorm,
            init_values: float | None = None,
            class_token: bool = True,
            num_reg_tokens: int = 0,
            no_embed_class: bool = False,
            use_abs_pos_emb: bool = True,
            use_rot_pos_emb: bool = False,
            rope_type: str | None = 'cat',
            rope_grid_offset: float = 0.,
            rope_grid_indexing: str = 'ij',
            rope_temperature: float = 10000.,
            rope_rotate_half: bool = False,
            use_post_norm: bool = False,
            use_pre_transformer_norm: bool = False,
            use_post_transformer_norm: bool | None = None,
            use_fc_norm: bool | None = None,
            attn_pool_num_heads: int | None = None,
            attn_pool_mlp_ratio: float | None = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            ref_feat_shape: tuple[int, int] | int | None = None,
            head_init_scale: float = 0.001,
    ) -> None:
        super(Eva, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            mlp_ratio=mlp_ratio,
            swiglu_mlp=swiglu_mlp,
            swiglu_align_to=swiglu_align_to,
            scale_mlp=scale_mlp,
            scale_attn_inner=scale_attn_inner,
            attn_type=attn_type,
            drop_rate=drop_rate,
            pos_drop_rate=pos_drop_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            class_token=class_token,
            num_reg_tokens=num_reg_tokens,
            no_embed_class=no_embed_class,
            use_abs_pos_emb=use_abs_pos_emb,
            use_rot_pos_emb=use_rot_pos_emb,
            rope_type=rope_type,
            rope_grid_offset=rope_grid_offset,
            rope_grid_indexing=rope_grid_indexing,
            rope_temperature=rope_temperature,
            rope_rotate_half=rope_rotate_half,
            use_post_norm=use_post_norm,
            use_pre_transformer_norm=use_pre_transformer_norm,
            use_post_transformer_norm=use_post_transformer_norm,
            use_fc_norm=use_fc_norm,
            attn_pool_num_heads=attn_pool_num_heads,
            attn_pool_mlp_ratio=attn_pool_mlp_ratio,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            ref_feat_shape=ref_feat_shape,
            head_init_scale=head_init_scale,
        )
    
    def get_arch(self) -> Literal['cnn', 'transformer']:
        return 'transformer'

    def forward_stem(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        x = self.norm_pre(x)
        return x

    def get_layers(self):
        return self.blocks

    def forward_pool(self, x: torch.Tensor):
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x

    def get_head(self):
        return self.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_stem(x)
        feats = {
            'feats': [],
            'preact_feats': [],
            'pooled_feat': None,
        }
        for block in self.blocks:
            x = block.forward(x)
            feats['preact_feats'].append(x)
            feats['feats'].append(x)
        x = self.forward_pool(x)
        feats['pooled_feat'] = x
        x = self.head(x)
        return x, feats

    def forward_partial(self, x: torch.Tensor, end_layer) -> torch.Tensor:
        x = self.forward_stem(x)
        feats = {
            'feats': [],
            'preact_feats': [],
            'pooled_feat': None,
        }
        for i, block in enumerate(self.blocks):
            if i > end_layer:
                return feats
            else:
                x = block.forward(x)
                feats['preact_feats'].append(x)
                feats['feats'].append(x)
        
        return feats

    def forward_wohead(self, x: torch.Tensor) ->torch.Tensor:
        """
        for distillate only feature, do not pass through pooling layer & head
        """
        x = self.forward_stem(x)
        feats = {
            'feats': [],
            'preact_feats': [],
            'pooled_feat': None,
        }
        for block in self.blocks:
            x = block.forward(x)
            feats['preact_feats'].append(x)
            feats['feats'].append(x)
        return x, feats


def _create_eva(variant: str, pretrained: bool = False, **kwargs) -> Eva:
    """Create an EVA model.

    Args:
        variant: Model variant name.
        pretrained: Load pretrained weights.
        **kwargs: Additional model arguments.

    Returns:
        Instantiated Eva model.
    """
    # Check if we should use NaFlexVit implementation
    use_naflex = kwargs.pop('use_naflex', None)
    _USE_NAFLEX_DEFAULT = os.environ.get('TIMM_USE_NAFLEX', '0') == '1'
    if use_naflex is None:
        use_naflex = _USE_NAFLEX_DEFAULT
    if use_naflex:
        # Import here to avoid circular import
        from .naflexvit import _create_naflexvit_from_eva
        return _create_naflexvit_from_eva(variant, pretrained, **kwargs)

    out_indices = kwargs.pop('out_indices', 3)
    model = build_model_with_cfg(
        Eva, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model


def vit_tiny_patch16_dinov3(pretrained: bool = False, **kwargs) -> Eva:
    """Custom, no pretraining"""
    model_args = dict(
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=192,
        depth=12,
        num_heads=6,
        qkv_bias=False,
        init_values=1.0e-05, # layer-scale
        rope_type='dinov3',
        rope_temperature=100,
        #rope_rescale_coords=2,  # haven't added to interface
        rope_rotate_half=True,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_layer=partial(LayerNorm, eps=1e-5),
    )
    model = _create_eva('vit_small_patch16_dinov3', pretrained=False, **dict(model_args, **kwargs))
    return model


def vit_small_patch16_dinov3(pretrained: bool = False, **kwargs) -> Eva:
    """DINOv3 S/16 https://arxiv.org/abs/2508.10104"""
    model_args = dict(
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=384,
        depth=12,
        num_heads=6,
        qkv_bias=False,
        init_values=1.0e-05, # layer-scale
        rope_type='dinov3',
        rope_temperature=100,
        #rope_rescale_coords=2,  # haven't added to interface
        rope_rotate_half=True,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_layer=partial(LayerNorm, eps=1e-5),
    )
    model = _create_eva('vit_small_patch16_dinov3', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_small_patch16_dinov3_qkvb(pretrained: bool = False, **kwargs) -> Eva:
    """DINOv3 S/16 w/ QKV bias enabled (but zero) https://arxiv.org/abs/2508.10104"""
    model_args = dict(
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=384,
        depth=12,
        num_heads=6,
        qkv_bias=True,
        init_values=1.0e-05, # layer-scale
        rope_type='dinov3',
        rope_temperature=100,
        #rope_rescale_coords=2,  # haven't added to interface
        rope_rotate_half=True,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_layer=partial(LayerNorm, eps=1e-5),
    )
    model = _create_eva('vit_small_patch16_dinov3_qkvb', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_small_plus_patch16_dinov3(pretrained: bool = False, **kwargs) -> Eva:
    """DINOv3 S/16 Plus https://arxiv.org/abs/2508.10104"""
    model_args = dict(
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=384,
        depth=12,
        num_heads=6,
        qkv_bias=False,
        init_values=1.0e-05, # layer-scale
        rope_type='dinov3',
        rope_temperature=100,
        #rope_rescale_coords=2,  # haven't added to interface
        rope_rotate_half=True,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        swiglu_mlp=True,
        swiglu_align_to=8,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_layer=partial(LayerNorm, eps=1e-5),
    )
    model = _create_eva('vit_small_plus_patch16_dinov3', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_small_plus_patch16_dinov3_qkvb(pretrained: bool = False, **kwargs) -> Eva:
    """DINOv3 S/16 Plus w/ QKV bias enabled (but 0) https://arxiv.org/abs/2508.10104"""
    model_args = dict(
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=384,
        depth=12,
        num_heads=6,
        qkv_bias=True,
        init_values=1.0e-05, # layer-scale
        rope_type='dinov3',
        rope_temperature=100,
        #rope_rescale_coords=2,  # haven't added to interface
        rope_rotate_half=True,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        swiglu_mlp=True,
        swiglu_align_to=8,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_layer=partial(LayerNorm, eps=1e-5),
    )
    model = _create_eva('vit_small_plus_patch16_dinov3_qkvb', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch16_dinov3(pretrained: bool = False, **kwargs) -> Eva:
    """DINOv3 B/16 https://arxiv.org/abs/2508.10104"""
    model_args = dict(
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_bias=False,
        init_values=1.0e-05, # layer-scale
        rope_type='dinov3',
        rope_temperature=100,
        #rope_rescale_coords=2,  # haven't added to interface
        rope_rotate_half=True,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_layer=partial(LayerNorm, eps=1e-5),
    )
    model = _create_eva('vit_base_patch16_dinov3', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch16_dinov3_qkvb(pretrained: bool = False, **kwargs) -> Eva:
    """DINOv3 B/16 w/ QKV bias enabled (but zero) https://arxiv.org/abs/2508.10104"""
    model_args = dict(
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_bias=True,
        init_values=1.0e-05, # layer-scale
        rope_type='dinov3',
        rope_temperature=100,
        #rope_rescale_coords=2,  # haven't added to interface
        rope_rotate_half=True,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_layer=partial(LayerNorm, eps=1e-5),
    )
    model = _create_eva('vit_base_patch16_dinov3_qkvb', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_large_patch16_dinov3(pretrained: bool = False, **kwargs) -> Eva:
    """DINOv3 L/16 https://arxiv.org/abs/2508.10104"""
    model_args = dict(
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        qkv_bias=False,
        init_values=1.0e-5, # layer-scale
        rope_type='dinov3',
        rope_temperature=100,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        rope_rotate_half=True,
        #rope_rescale_coords=2,  # haven't added to interface
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_layer=partial(LayerNorm, eps=1e-5),
    )
    model = _create_eva('vit_large_patch16_dinov3', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_large_patch16_dinov3_qkvb(pretrained: bool = False, **kwargs) -> Eva:
    """DINOv3 w/ QKV bias enabled (but zero) https://arxiv.org/abs/2508.10104"""
    model_args = dict(
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        qkv_bias=True,
        init_values=1.0e-5, # layer-scale
        rope_type='dinov3',
        rope_temperature=100,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        rope_rotate_half=True,
        #rope_rescale_coords=2,  # haven't added to interface
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_layer=partial(LayerNorm, eps=1e-5),
    )
    model = _create_eva('vit_large_patch16_dinov3_qkvb', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_huge_plus_patch16_dinov3(pretrained: bool = False, **kwargs) -> Eva:
    """DINOv3 H/16 Plus https://arxiv.org/abs/2508.10104"""
    model_args = dict(
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        qkv_bias=False,
        init_values=1.0e-5, # layer-scale
        rope_type='dinov3',
        rope_temperature=100,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        rope_rotate_half=True,
        swiglu_mlp=True,
        swiglu_align_to=8,
        #rope_rescale_coords=2,  # haven't added to interface
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_layer=partial(LayerNorm, eps=1e-5),
    )

    model = _create_eva('vit_huge_plus_patch16_dinov3', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_huge_plus_patch16_dinov3_qkvb(pretrained: bool = False, **kwargs) -> Eva:
    """DINOv3 H/16 Plus w/ QKV bias enabled (but zero) https://arxiv.org/abs/2508.10104"""
    model_args = dict(
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        qkv_bias=True,
        init_values=1.0e-5, # layer-scale
        rope_type='dinov3',
        rope_temperature=100,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        rope_rotate_half=True,
        swiglu_mlp=True,
        swiglu_align_to=8,
        #rope_rescale_coords=2,  # haven't added to interface
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_layer=partial(LayerNorm, eps=1e-5),
    )

    model = _create_eva('vit_huge_plus_patch16_dinov3_qkvb', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_7b_patch16_dinov3(pretrained: bool = False, **kwargs) -> Eva:
    """DINOv3 7B/16 https://arxiv.org/abs/2508.10104"""
    model_args = dict(
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        qkv_bias=False,
        mlp_ratio=2,
        init_values=1.0e-5, # layer-scale
        rope_type='dinov3',
        rope_temperature=100,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        rope_rotate_half=True,
        swiglu_mlp=True,
        swiglu_align_to=64,
        #rope_rescale_coords=2,  # haven't added to interface
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_layer=partial(LayerNorm, eps=1e-5),
    )

    model = _create_eva('vit_7b_patch16_dinov3', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

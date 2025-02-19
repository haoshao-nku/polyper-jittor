_base_ = [
    '../_base_/datasets/polyp_512x512.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]


# model settings
norm_cfg = dict(type='BN')
backbone_norm_cfg = dict(type='LN')
model = dict(
    type='EncoderDecoder',
    pretrained='jittorhub://swin_tiny_patch4_window7_224.pkl',
    backbone=dict(type='SwinTransformer',
                  pretrain_img_size=224,
                  embed_dims=96,
                  patch_size=4,
                  window_size=7,
                  mlp_ratio=4,
                  depths=[2, 2, 6, 2],
                  num_heads=[3, 6, 12, 24],
                  strides=(4, 2, 2, 2),
                  out_indices=(0, 1, 2, 3),
                  qkv_bias=True,
                  qk_scale=None,
                  patch_norm=True,
                  drop_rate=0.,
                  attn_drop_rate=0.,
                  drop_path_rate=0.3,
                  use_abs_pos_embed=False,
                  act_cfg=dict(type='GELU'),
                  norm_cfg=backbone_norm_cfg),  
    decode_head=dict(
        type='Polyper',
        in_channels=[96,192,384,768],
        channels=96,
        image_size = 128,
        in_index=[0, 1, 2, 3],
        heads= 8,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
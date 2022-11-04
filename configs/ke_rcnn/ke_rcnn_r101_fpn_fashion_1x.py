_base_ = [
    './ke_rcnn_r50_fpn_fashion_1x.py',
]

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

# fp16
fp16 = dict(loss_scale=dict(init_scale=512.))
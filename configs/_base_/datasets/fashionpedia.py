# dataset settings
classes = ('shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan',
           'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit',
           'cape', 'glasses', 'hat', 'headband, head covering, hair accessory',
           'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings',
           'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar',
           'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 
           'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon',
           'rivet', 'ruffle', 'sequin', 'tassel', 'human')

dataset_type = 'FashionPedia'
data_root = 'data/fashionpedia/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadFashionAnnotations',
         with_bbox=True,
         with_mask=True,
         with_attribute=True,
         with_human=True),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1,1)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 
                               'gt_bboxes', 
                               'gt_labels',
                               'gt_attributes',
                               'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'instances_attributes_train2020.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline,
        with_human=True),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'instances_attributes_val2020.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline,
        with_human=True),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'instances_attributes_val2020.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline,
        with_human=True))

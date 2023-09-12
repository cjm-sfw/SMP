nums_gpu = 2
batch_size_per_card = 4
# model settings
model = dict(
    type='SingleStageInsParsingDetector',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
        style='pytorch',
        dcn=dict(
            type='DCNv2',
            deformable_groups=1,
            fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    mask_head=dict(
        type='Repparsing_Head',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        seg_feat_channels=128,
        strides=[32],
        scale_ranges=((1, 2048)),
        sigma=0.1,
        num_grid=36,
        ins_out_channels=256,
        enable_ori_grid= True, #['cate', 'kernel', False]
        enable_center=True,
        enable_kernel=False,
        enable_instance=True,
        enable_semantic=True,
        enable_fusion=False,
        enable_fusion_multi=True,
        enable_metrics=True,
        enable_moi = False,
        enable_adapt_center = True,
        enable_diceloss_weight = False,
        parsing2mask = True,
        mask_feature_head=dict(
            in_channels=256,
            feat_channels=128,
            start_level=0,
            end_level=3,
            out_channels=256,
            mask_stride=4,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        loss_ins=dict(
            type='DiceLoss',
            use_sigmoid=True,
            loss_weight=3.0),
        loss_semantic=dict(
            type='DiceLoss',
            use_sigmoid=True,
            loss_weight=3.0),
        loss_parsing=dict(
            type='DiceLoss',
            use_sigmoid=True,
            loss_weight=3.0),
        loss_metrics=dict(
            type='Similarityloss',
            metric_type='cosine',
            contend=['semantic_kernels', 
            'instance_kernels',
            #'parsing_kernels'
            'parsing_features'
                    ],
            margin=0.04,
            loss_weight=4.0),
        loss_semantic_neg=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_semantic_neg_config=dict(
            enable=False,
            loss_semantic_pos_square=False,
            loss_semantic_neg_l2=True,
        ),
        loss_center=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)),
    train_cfg = dict(),
    test_cfg = dict(
        nms_pre=500,
        ctr_score=0.1,
        score_thr=0.1,
        cate_score_thr=0.3,
        mask_thr=0.5,
        cate_update_thr=0.1,
        update_thr=0.1,
        kernel='gaussian',  # gaussian/linear
        sigma=2.0,
        max_per_img=100,
        debug_heatmap=False)
    )
# training and testing settings
# train_cfg = dict()
# test_cfg = dict(
#     nms_pre=500,
#     ctr_score=0.1,
#     score_thr=0.1,
#     cate_score_thr=0.3,
#     mask_thr=0.5,
#     cate_update_thr=0.1,
#     update_thr=0.1,
#     kernel='gaussian',  # gaussian/linear
#     sigma=2.0,
#     max_per_img=100,
#     debug_heatmap=False)
# dataset settings
dataset_type = 'DensePose'
data_root = '/root/autodl-tmp/Densepose_CoCo/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile_Parsing'),
    dict(type='LoadAnnotations_Parsing', with_bbox=True, with_mask=True, with_parsing=True),
    dict(type='Resize_Parsing',
         img_scale=[(1333, 800), (1333, 768), (1333, 736),
                    (1333, 704), (1333, 672), (1333, 640)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip_Parsing', flip_ratio=0.5, dataset='densepose'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad_Parsing', size_divisor=32),
    dict(type='DefaultFormatBundle_Parsing'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_parsing'],
                        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg')),
    ]
test_pipeline = [
    dict(type='LoadImageFromFile_Parsing'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize_Parsing', keep_ratio=True),
            dict(type='RandomFlip_Parsing', dataset='densepose'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad_Parsing', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg')),
        ])
]
data = dict(
    samples_per_gpu=batch_size_per_card,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/densepose_coco_2014_train.json',
        img_prefix=data_root + 'images/train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/densepose_coco_2014_minival.json',
        img_prefix=data_root + 'images/val2017/',
        test_mode = True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/densepose_coco_2014_minival.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline))

# optimizer
lr_base = 0.000625
optimizer = dict(type='SGD', lr=(lr_base*data['samples_per_gpu'] * nums_gpu), momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=optimizer['lr'],
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=6,
                        create_symlink=False)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/densepose_release_r101_fpn_8gpu_3x_offset_parsing_v1/noneg_cluster/'
load_from = ''
resume_from = ''
workflow = [('train', 1)]

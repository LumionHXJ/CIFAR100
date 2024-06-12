_base_ = [
    'mmpretrain::convnext_v2/convnext-v2-base_32xb32_in1k.py',
] 

model = dict(
    _delete_ = True,
    type = 'TimmClassifier',
    init_cfg=dict(
        bias=0.0, layer=[
            'Conv2d',
            'Linear',
        ], std=0.02, type='TruncNormal'),
    train_cfg=dict(augments=[
        dict(alpha=0.8, type='Mixup'),
        dict(alpha=1.0, type='CutMix'),
    ]),
    loss=dict(
        type='TIMM_LabelSmoothingCrossEntropy', 
        label_smooth=0.1,
        num_classes=100),
    data_preprocessor = dict(
        num_classes=100,
        # RGB format normalization parameters
        mean=[129.304, 124.070, 112.434],
        std=[68.170, 65.392, 70.418],
        # loaded images are already RGB format
        to_rgb=False),
    model_name='convnext_base',
    pretrained=False,
    num_classes=100,
    use_grn=True,
    drop_path_rate=0.1,
    ls_init_value=0.,
    kernel_sizes=3,
)

train_pipeline = [
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=256,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='TIMM_AutoAugment'),
    dict(
        erase_prob=0.25,
        fill_color=[
            103.53,
            116.28,
            123.675,
        ],
        fill_std=[
            57.375,
            57.12,
            58.395,
        ],
        max_area_ratio=0.3333333333333333,
        min_area_ratio=0.02,
        mode='rand',
        type='RandomErasing'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(
        type='Resize',
        scale=256,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        type='CIFAR100',
        data_root='data/',
        download=True,
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        type='CIFAR100',
        data_root='data/',
        split='test',
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(type='Accuracy', topk=(1, ))
test_evaluator = val_evaluator

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])

optim_wrapper = dict(
    type='TIMM_AmpOptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW', 
        lr=1e-4, 
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999))
)

default_hooks = dict(
    checkpoint=dict(interval=2, type='CheckpointHook')
)

train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=2)
auto_scale_lr = dict(base_batch_size=128)

logger=dict(interval=10, type='LoggerHook')

param_scheduler = [dict(type='ReduceOnPlateauLR', 
                         rule='greater',
                         monitor='accuracy/top1',
                         factor=0.1, 
                         patience=1,
                         threshold=1e-4)]

custom_imports = dict(
    imports=['modules'], 
    allow_failed_imports=False)

del _base_.custom_hooks

resume = False
load_from = 'work_dirs_base/epoch_300.pth'
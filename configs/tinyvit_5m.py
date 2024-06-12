_base_ = ['mmpretrain::tinyvit/tinyvit-5m_8xb256_in1k.py']

data_preprocessor = dict(
    num_classes=100,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

model = dict(
    backbone=dict(
        img_size=(64, 64),
        window_size=[3, 3, 6, 3],
    ),
    head=dict(
        num_classes=100,
        loss=dict(
            _delete_=True,
            type='TIMM_LabelSmoothingCrossEntropy', 
            label_smooth=0.1,
            num_classes=100)
    ),
    data_preprocessor=data_preprocessor,
    train_cfg=dict(augments=[
        dict(alpha=0.8, type='Mixup'),
        dict(alpha=1.0, type='CutMix'),
    ]),
)

train_pipeline = [
    dict(
        type='RandomResizedCrop',
        scale=64,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
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
        type='ResizeEdge',
        scale=64,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=64),
    dict(type='PackInputs'),
]


train_dataloader = dict(
    batch_size=64,
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
    batch_size=64,
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


default_hooks = dict(
    checkpoint=dict(interval=5, type='CheckpointHook')
)

train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=5)
auto_scale_lr = dict(base_batch_size=128)

logger=dict(interval=10, type='LoggerHook')

custom_imports = dict(
    imports=['modules'], 
    allow_failed_imports=False)

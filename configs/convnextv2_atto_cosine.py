_base_ = [
    './convnextv2_base.py'
]

model = dict(
    model_name='convnext_atto',
)

param_scheduler = [
    dict(
        _scope_='mmpretrain',
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        start_factor=0.001,
        type='LinearLR'),
    dict(
        _scope_='mmpretrain',
        begin=20,
        by_epoch=True,
        eta_min=1e-05,
        type='CosineAnnealingLR'),
]
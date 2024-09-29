_base_ = '../yolox/yolox_s_8xb8-300e_coco.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=0.67, widen_factor=0.75),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
    bbox_head=dict(num_classes=5,in_channels=192, feat_channels=192),
)

data_root = '/home/data2/lxm/datasets/SIXray_coco/'

metainfo = {
    'classes': ('Gun','Knife','Wrench','Pliers','Scissors')
}

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    dataset=dict(
        data_root=data_root,
        ann_file='/home/data2/lxm/datasets/SIXray_coco/annotations/instances_train2017.json',
        data_prefix=dict(img='/home/data2/lxm/datasets/SIXray_coco/train2017/')
    ))
train_dataloader = dict(
    num_workers=8,
    batch_size=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_dataloader = dict(
    num_workers=8,
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/home/data2/lxm/datasets/SIXray_coco/annotations/instances_val2017.json',
        data_prefix=dict(img='/home/data2/lxm/datasets/SIXray_coco/val2017/')))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

max_epochs = 32
num_last_epochs = 5
interval = 4
base_lr = 0.01

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]
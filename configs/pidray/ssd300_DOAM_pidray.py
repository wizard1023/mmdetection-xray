_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
input_size = 300
model = dict(
    backbone=dict(
            type='SSDVGG_DOAM',
            depth=16,
            with_last_pool=False,
            ceil_mode=True,
            out_indices=(3, 4),
            out_feature_indices=(22, 34),
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://vgg16_caffe')),
    bbox_head=dict(
            type='SSDHead',
            in_channels=(512, 1024, 512, 256, 256, 256),
            num_classes=12,
            anchor_generator=dict(
                type='SSDAnchorGenerator',
                scale_major=False,
                input_size=input_size,
                basesize_ratio_range=(0.15, 0.9),
                strides=[8, 16, 32, 64, 100, 300],
                ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2])),
    test_cfg=dict(
            nms_pre=1000,
            nms=dict(type='nms', iou_threshold=0.45),
            min_bbox_size=0,
            score_thr=0.10,
            max_per_img=200)
)

# dataset settings
data_root = '/home/xray/HYB/data/PIDray/'

metainfo = {
    'classes': ('Gun','Bullet','Knife','Wrench','Pliers','Powerbank','Baton','Lighter','Sprayer',
                'Hammer','Scissors','Handcuffs')
}

train_dataloader = dict(
    num_workers=8,
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/home/xray/HYB/data/PIDray/annotations/instances_train2017.json',
        data_prefix=dict(img='/home/xray/HYB/data/PIDray/train2017/')))
val_dataloader = dict(
    num_workers=8,
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/home/xray/HYB/data/PIDray/annotations/instances_val2017.json',
        data_prefix=dict(img='/home/xray/HYB/data/PIDray/val2017/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4))

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

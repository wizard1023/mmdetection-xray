_base_ = [
    '../_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        bbox_head=[
                    dict(
                        type='Shared2FCBBoxHead',
                        in_channels=256,
                        fc_out_channels=1024,
                        roi_feat_size=7,
                        num_classes=5,
                        bbox_coder=dict(
                            type='DeltaXYWHBBoxCoder',
                            target_means=[0., 0., 0., 0.],
                            target_stds=[0.1, 0.1, 0.2, 0.2]),
                        reg_class_agnostic=True,
                        loss_cls=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=1.0),
                        loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                       loss_weight=1.0)),
                    dict(
                        type='Shared2FCBBoxHead',
                        in_channels=256,
                        fc_out_channels=1024,
                        roi_feat_size=7,
                        num_classes=5,
                        bbox_coder=dict(
                            type='DeltaXYWHBBoxCoder',
                            target_means=[0., 0., 0., 0.],
                            target_stds=[0.05, 0.05, 0.1, 0.1]),
                        reg_class_agnostic=True,
                        loss_cls=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=1.0),
                        loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                       loss_weight=1.0)),
                    dict(
                        type='Shared2FCBBoxHead',
                        in_channels=256,
                        fc_out_channels=1024,
                        roi_feat_size=7,
                        num_classes=5,
                        bbox_coder=dict(
                            type='DeltaXYWHBBoxCoder',
                            target_means=[0., 0., 0., 0.],
                            target_stds=[0.033, 0.033, 0.067, 0.067]),
                        reg_class_agnostic=True,
                        loss_cls=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=1.0),
                        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
                ],
        mask_head=dict(
                    type='FCNMaskHead',
                    num_convs=4,
                    in_channels=256,
                    conv_out_channels=256,
                    num_classes=5,
                    loss_mask=dict(
                        type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)))
)

data_root = '/home/data2/lxm/datasets/SIXray_coco/'

metainfo = {
    'classes': ('Gun','Knife','Wrench','Pliers','Scissors')
}

train_dataloader = dict(
    num_workers=8,
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/home/data2/lxm/datasets/SIXray_coco/annotations/instances_train2017_mask.json',
        data_prefix=dict(img='/home/data2/lxm/datasets/SIXray_coco/train2017/')))
val_dataloader = dict(
    num_workers=8,
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/home/data2/lxm/datasets/SIXray_coco/annotations/instances_val2017_mask.json',
        data_prefix=dict(img='/home/data2/lxm/datasets/SIXray_coco/val2017/')))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017_mask.json')
test_evaluator = val_evaluator
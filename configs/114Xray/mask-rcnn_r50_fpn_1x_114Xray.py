_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=12), mask_head=dict(num_classes=12)))

data_root = '/home/xray/HYB/mmdetection_xray/data/114Xray_contrast_1/'

metainfo = {
    'classes': ('powerbank', 'mobilephone', 'battery', 'scissors', 'suspiciousliquid1',
                'suspiciousliquid2', 'fruitknife', 'cleaver', 'lighter', 'handcuffs', 'expandablebatons', 'pressure')
}

train_dataloader = dict(
    num_workers=8,
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/home/xray/HYB/mmdetection_xray/data/114Xray_contrast_1/annotations/instances_train2017_mask.json',
        data_prefix=dict(img='/home/xray/HYB/mmdetection_xray/data/114Xray_contrast_1/train2017/')))
val_dataloader = dict(
    num_workers=8,
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/home/xray/HYB/mmdetection_xray/data/114Xray_contrast_1/annotations/instances_val2017_mask.json',
        data_prefix=dict(img='/home/xray/HYB/mmdetection_xray/data/114Xray_contrast_1/val2017/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017_mask.json')
test_evaluator = val_evaluator
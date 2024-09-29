_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=8), mask_head=dict(num_classes=8)))

data_root = '/home/xray/HYB/data/HiXray/'

metainfo = {
    'classes': ('Mobile_Phone','Laptop','Portable_Charger_1','Portable_Charger_2','Tablet','Cosmetic','Water','Nonmetallic_Lighter')
}

train_dataloader = dict(
    num_workers=8,
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/home/xray/HYB/data/HiXray/annotations/instances_train2017_mask.json',
        data_prefix=dict(img='/home/xray/HYB/data/HiXray/train2017/')))
val_dataloader = dict(
    num_workers=8,
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/home/xray/HYB/data/HiXray/annotations/instances_val2017_mask.json',
        data_prefix=dict(img='/home/xray/HYB/data/HiXray/val2017/')))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017_mask.json')
test_evaluator = val_evaluator
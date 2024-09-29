_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    '../retinanet/retinanet_tta.py'
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

model = dict(
        bbox_head=dict(num_classes=5))

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
        ann_file='/home/data2/lxm/datasets/SIXray_coco/annotations/instances_train2017.json',
        data_prefix=dict(img='/home/data2/lxm/datasets/SIXray_coco/train2017/')))
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
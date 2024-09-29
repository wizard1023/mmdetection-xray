from mmdet.apis import DetInferencer
inferencer = DetInferencer(model='/home/data2/lxm/mmdetection/work_dirs/sixray/atss_r50_fpn_1x_sixray/atss_r50_fpn_1x_sixray.py',
                           weights='/home/data2/lxm/mmdetection/work_dirs/sixray/atss_r50_fpn_1x_sixray/epoch_32.pth',device='cuda:1')
inferencer('/home/data2/lxm/datasets/SIXray_coco/val2017/', out_dir='/home/data2/lxm/datasets/SIXray_coco/vis_atss/', no_save_pred=False)

from mmdet.apis import DetInferencer

imagepath = '/home/xray/HYB/data/SIXray/val2017'  # 需要加载的测试图片的文件路径
# /home/xray/LXM/mmdetection/demo/P00170.jpg
# /home/xray/HYB/data/SIXray/val2017
savepath = '/home/data2/lxm/mmdetection/pred_visual/cascade-rcnn-res50-fpn/sixray'  # 保存测试图片的路径
# /home/xray/LXM/mmdetection/bbox_visual
# /home/data2/lxm/mmdetection/pred_visual/cascade-rcnn-res50-fpn/sixray
config_file = '/home/xray/LXM/mmdetection/configs/sixray/cascade-rcnn_r50_fpn_1x_sixray.py'  # 网络模型
# /home/xray/LXM/mmdetection/configs/sixray/cascade-rcnn_r50_edge_material_8_fpn_1x_sixray.py

checkpoint_file = '/home/data2/lxm/mmdetection/work_dirs/cascade-rcnn_r50_fpn_1x_sixray/epoch_36.pth'  # 训练好的模型参数
# /home/data2/lxm/mmdetection/work_dirs/cascade-rcnn_r50_edge_material_8_fpn_1x_sixray/epoch_18.pth
device = 'cuda:1'
# init a detector
inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device=device)
# inference the demo image

inferencer(imagepath, out_dir=savepath, no_save_pred=True)
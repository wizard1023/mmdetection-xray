import json
import os
import random

import cv2

root_path = '/home/xray/HYB/data/SIXray'

id_category = {1:'Gun',2:'Knife',3:'Wrench',4:'Pliers',5:'Scissors'}  # 改成自己的类别
# id_category = {1:'Folding_Knife',2:'Straight_Knife',3:'Multi-tool_Knife',4:'Utility_Knife',5:'Scissor'}
colors={
    1:(133,21,199),
    2:(0,0,139),
    3:(225,105,65),
    4:(0,140,255),
    5:(0,215,255)
}
def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=0.7,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness)

    return text_size

def visiual():
    # 获取bboxes
    json_file = os.path.join(root_path, 'annotations', 'instances_train2017.json')  # 如果想查看验证集，就改这里
    data = json.load(open(json_file, 'r'))
    images = data['images']  # json中的image列表，

    # 读取图片
    for i in images:  # 随机挑选SAMPLE_NUMBER个检测
        # for i in images:                                        # 整个数据集检查
        img = cv2.imread(os.path.join(root_path, 'train2017',
                                      i['file_name']))  # 改成验证集的话，这里的图片目录也需要改,train2017 -> val2017
        bboxes = []  # 获取每个图片的bboxes
        category_ids = []
        annotations = data['annotations']
        for j in annotations:
            if j['image_id'] == i['id']:
                bboxes.append(j["bbox"])
                category_ids.append(j['category_id'])

        # 生成锚框
        for idx, bbox in enumerate(bboxes):

            left_top = (int(bbox[0]), int(bbox[1]))  # 这里数据集中bbox的含义是，左上角坐标和右下角坐标。
            right_bottom = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))  # 根据不同数据集中bbox的含义，进行修改。
            color = colors[category_ids[idx]]
            cv2.rectangle(img, left_top, right_bottom, color=color, thickness=2)  # 图像，左上角，右下坐标，颜色，粗细
            draw_text(img, id_category[category_ids[idx]], pos=left_top, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, text_color=(200, 200, 200), font_thickness=1)
            # 画出每个bbox的类别，参数分别是：图片，类别名(str)，坐标，字体，大小，颜色，粗细
        # cv2.imshow('image', img)                                          # 展示图片，
        # cv2.waitKey(1000)
        cv2.imwrite(os.path.join('/home/data2/lxm/datasets/SIXray_coco/vis_edge_material_10', i['file_name']), img)  # 或者是保存图片
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    print('—' * 50)
    visiual()
    print('| visiual completed.')
    print('—' * 50)

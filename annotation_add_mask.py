import json
from pprint import pprint
def convert_bbox_to_polygon(bbox):
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    polygon = [x,y,(x+w),y,(x+w),(y+h),x,(y+h)]
    return([polygon])
def main():
    # file_path = "/home/xray/HYB/data/HiXray/annotations/instances_val2017.json"
    file_path = "/home/xray/HYB/mmdetection_xray/data/114Xray_contrast_1/annotations/instances_train2017.json"
    f = open(file_path)
    data = json.load(f)
    for line in data["annotations"]:
        segmentation = convert_bbox_to_polygon(line["bbox"])
        line["segmentation"] = segmentation
    with open("/home/xray/HYB/mmdetection_xray/data/114Xray_contrast_1/annotations/instances_train2017_mask.json", 'w') as f:
        f.write(json.dumps(data))
    print('DONE')
main()
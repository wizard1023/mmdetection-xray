import cv2
img = cv2.imread('/home/xray/LXM/mmdetection/demo/P08844.jpg')
# B, G, R = cv2.split(img)  # 分离出图片的B，R，G颜色通道
# cv2.imwrite("/home/xray/LXM/mmdetection/demo/006619401009995_RED.jpg", R) # 显示三通道的值都为R值时d图片
# cv2.imwrite("/home/xray/LXM/mmdetection/demo/006619401009995_GREEN.jpg", G)  # 显示三通道的值都为G值时d图片
# cv2.imwrite("/home/xray/LXM/mmdetection/demo/006619401009995_BLUE.jpg", B)  # 显示三通道的值都为B值时d图片
# img_mean_B = cv2.medianBlur(B, 5)
# img_mean_G = cv2.medianBlur(G, 5)
# img_mean_R = cv2.medianBlur(R, 5)
# cv2.imwrite("/home/xray/LXM/mmdetection/demo/006619401009995_RED_medianBlur.jpg", img_mean_R) # 显示三通道的值都为R值时d图片
# cv2.imwrite("/home/xray/LXM/mmdetection/demo/006619401009995_GREEN_medianBlur.jpg", img_mean_G)  # 显示三通道的值都为G值时d图片
# cv2.imwrite("/home/xray/LXM/mmdetection/demo/006619401009995_BLUE_medianBlur.jpg", img_mean_B)  # 显示三通道的值都为B值时d图片

# rgb转hsv
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
img_rgb_hsv = 0.7 * img + 0.3 * img_hsv
h,s,v = cv2.split(img_hsv)
# cv2.imwrite('/home/xray/LXM/mmdetection/demo/P00151_hsv.jpg',img_hsv)
# cv2.imwrite('/home/xray/LXM/mmdetection/demo/P00151_rgb_hsv.jpg',img_rgb_hsv)
# cv2.imwrite('/home/xray/LXM/mmdetection/demo/P00087_h.jpg',h)
# cv2.imwrite('/home/xray/LXM/mmdetection/demo/P00087_s.jpg',s)
# cv2.imwrite('/home/xray/LXM/mmdetection/demo/P00087_v.jpg',v)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_th = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
_,img_otsu = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('/home/xray/LXM/mmdetection/demo/P08844_gray.jpg',gray_image)
cv2.imwrite('/home/xray/LXM/mmdetection/demo/P08844_th.jpg',img_th)
cv2.imwrite('/home/xray/LXM/mmdetection/demo/P08844_otsu.jpg',img_otsu)
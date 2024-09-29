import cv2

# 图片的共同前缀
# prefix = 'P07673'
# prefix = 'P07122'
prefix = 'P03330'
# 手动裁剪第一张图片
img1 = cv2.imread(f'C:\\Users\\LHY\\Desktop\\sixray\\gt\\{prefix}.jpg')
x, y, w, h = cv2.selectROI(img1)
crop_img1 = img1[y:y+h, x:x+w]
cv2.imwrite(f'result/{prefix}_gt.jpg', crop_img1)

# 对第二张图片做相同的裁剪
img2 = cv2.imread(f'C:\\Users\\LHY\\Desktop\\sixray\\vis\\{prefix}.jpg')
crop_img2 = img2[y:y+h, x:x+w]
cv2.imwrite(f'result/{prefix}_vis.jpg', crop_img2)

# 对第三张图片做相同的裁剪
img3 = cv2.imread(f'C:\\Users\\LHY\\Desktop\\sixray\\vis_cascade\\{prefix}.jpg')
crop_img3 = img3[y:y+h, x:x+w]
cv2.imwrite(f'result/{prefix}_vis_cascade.jpg', crop_img3)

img4 = cv2.imread(f'C:\\Users\\LHY\\Desktop\\sixray\\vis_atss\\{prefix}.jpg')
crop_img4 = img4[y:y+h, x:x+w]
cv2.imwrite(f'result/{prefix}_vis_atss.jpg', crop_img4)

img5 = cv2.imread(f'C:\\Users\\LHY\\Desktop\\sixray\\vis_sdanet\\{prefix}.jpg')
crop_img5 = img5[y:y+h, x:x+w]
cv2.imwrite(f'result/{prefix}_vis_sdanet.jpg', crop_img5)

# img1 = cv2.imread(r'C:\Users\LHY\Desktop\sixray\gt\P00174.jpg')
# x, y, w, h = cv2.selectROI(img1)
# crop_img1 = img1[y:y+h, x:x+w]
# cv2.imwrite(f'result/{prefix}_crop6.jpg', crop_img1)

# img2 = cv2.imread(f'C:\\Users\\LHY\\Desktop\\sixray\\gt\\P00467.jpg')
# x, y, w, h = cv2.selectROI(img2)
# crop_img2 = img2[y:y+h, x:x+w]
# cv2.imwrite(f'result/{prefix}_crop2.jpg', crop_img2)
#
# img3 = cv2.imread(f'C:\\Users\\LHY\\Desktop\\sixray\\gt\\P01920.jpg')
# x, y, w, h = cv2.selectROI(img3)
# crop_img3 = img3[y:y+h, x:x+w]
# cv2.imwrite(f'result/{prefix}_crop3.jpg', crop_img3)



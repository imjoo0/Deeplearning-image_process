import cv2

img = cv2.imread('01.jpeg')

print(img)
print(img.shape) # (404, 640, 3) = (높이, 너비, 채널)

cv2.rectangle(img, pt1=(259, 89), pt2=(380, 348), color=(255, 0, 0), thickness=2)

cv2.imshow('img',img)
cv2.waitKey(0)

cropped_img = img[89:348, 259:380] 

cv2.imshow('cropped', cropped_img)
cv2.waitKey(0)

img_resized = cv2.resize(img, (512, 256))

cv2.imshow('resized', img_resized)
cv2.waitKey(0)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('result', img_rgb)
cv2.waitKey(0)

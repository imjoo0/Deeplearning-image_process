import cv2
import numpy as np 

#고흐의 빛나는 밤 로드 
net = cv2.dnn.readNetFromTorch('models/eccv16/starry_night.t7')

img = cv2.imread('imgs/01.jpeg')

h, w, c = img.shape

img = cv2.resize(img, dsize=(500, int(h / w * 500)))
print(img.shape) #blob 전처리 어떻게 변하는 지 확인하는 디버깅 코드 

MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)
print(blob.shape) #blob 전처리 어떻게 변하는 지 확인하는 디버깅 코드

net.setInput(blob)
output = net.forward()

output = output.squeeze().transpose((1, 2, 0)) # 늘렸던 차원을 다시 squeeze로 줄여 주고 , transpose (차원 변형했던것을 다시 돌려놓음)
output += MEAN_VALUE

output = np.clip(output, 0, 255)
output = output.astype('uint8')

#액자 부분 잘라서 다른 화풍으로 변형하기 

net2 = cv2.dnn.readNetFromTorch('models/instance_norm/mosaic.t7')

img2 = cv2.imread('imgs/02.jpg')

h, w, c = img2.shape

img2 = cv2.resize(img2, dsize=(500, int(h / w * 500)))

img2 = img2[162:513, 185:428]

MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

net2.setInput(blob)
output2 = net2.forward()

output2 = output2.squeeze().transpose((1, 2, 0)) # 늘렸던 차원을 다시 squeeze로 줄여 주고 , transpose (차원 변형했던것을 다시 돌려놓음)
output2 += MEAN_VALUE

output2 = np.clip(output2, 0, 255)
output2 = output2.astype('uint8')

cv2.imshow('img', img)
cv2.imshow('result', output)

cv2.imshow('img', img2)
cv2.imshow('result', output2)
cv2.waitKey(0)

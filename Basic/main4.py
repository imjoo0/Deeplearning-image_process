import cv2

cap = cv2.VideoCapture('04.mp4')
# cap = cv2.VideoCapture(0) # 컴퓨터 웹캠으로 틀고 싶을떼, 

while True:
	ret, img = cap.read()

	if ret == False:
		break

    cv2.rectangle(img, pt1=(721, 183), pt2=(878, 465), color=(255, 0, 0), thickness=2)  
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, dsize=(640, 360))
    
    img = img[100:200, 150:250]
    
	cv2.imshow('result', img)
	
	if cv2.waitKey(1) == ord('q'):
		break
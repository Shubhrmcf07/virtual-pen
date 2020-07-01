import cv2
import numpy as np 
import time

cap = cv2.VideoCapture(0)

pen_img = cv2.resize(cv2.imread('pen.png',1), (50, 50))
eraser_img = cv2.resize(cv2.imread('eraser.jpg',1), (50, 50))

cap.set(3, 1280)
cap.set(4,720)

kernel = np.ones((5,5), np.uint8)

canvas = None

x1,y1 = 0,0

noiseth = 200

backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

background_threshold = 600

switch = 'Pen'

last_switch = time.time()

wiper_thresh = 40000

clear = False

while True:
	_, frame = cap.read()

	frame = cv2.flip(frame, 1)

	if canvas is None:
		canvas = np.zeros_like(frame)

	top_left = frame[0:50, 0:50]
	fgmask = backgroundobject.apply(top_left)

	switch_thresh = np.sum(fgmask==255)

	if switch_thresh>background_threshold and (time.time()-last_switch) > 1:
	    last_switch = time.time()
	    if switch == 'Pen':
	    	switch = 'Eraser'
	    else:
	    	switch = 'Pen'
	 

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_range = np.array([97, 144, 107])
	upper_range = np.array([179, 255, 255])

	mask = cv2.inRange(hsv, lower_range, upper_range)

	mask = cv2.erode(mask, kernel, iterations=2)
	mask = cv2.dilate(mask, kernel, iterations=2)

	contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

	if contours and cv2.contourArea(max(contours,key = cv2.contourArea)) > noiseth:
		c = max(contours, key=cv2.contourArea)
		x2,y2,w,h = cv2.boundingRect(c)

		area = cv2.contourArea(c)

		if x1==0 and y1==0:
			x1,y1=x2,y2

		else:
			if switch == 'Pen':
				canvas = cv2.line(canvas, (x1,y1),(x2,y2), [255,0,0], 5)
			else:
				cv2.circle(canvas, (x2, y2), 20, (0,0,0), -1)

		x1,y1 = x2,y2

		if area > wiper_thresh:
			cv2.putText(canvas, 'Clearing canvas', (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
			clear = True


	else:
		x1,y1 = 0,0
	_ , mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
	foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
	background = cv2.bitwise_and(frame, frame,mask = cv2.bitwise_not(mask))
	frame = cv2.add(foreground,background)



	res = cv2.bitwise_and(frame, frame, mask=mask)

	mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

	stacked = np.hstack((mask_3, frame, res))

	if switch != 'Pen':
		cv2.circle(frame, (x1, y1), 10, (255,255,255), -1)
		frame[0: 50, 0: 50] = eraser_img
	else:
	    frame[0: 50, 0: 50] = pen_img

	# cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.4, fy=0.4))
	cv2.imshow('frame', frame)

	k = cv2.waitKey(5) & 0xFF

	if k==27:
		break

	if k==ord('c'):
		canvas = None
		clear = False

	if clear == True:
		time.sleep(1)
		canvas = None
		clear = False

cap.release()
cv2.destroyAllWindows()


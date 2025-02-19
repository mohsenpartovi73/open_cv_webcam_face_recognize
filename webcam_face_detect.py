
import cv2
import numpy
import os 

haar_file = '/home/mohsen/Documents/trainee/tutorial/open_cv/face_recognition/haar_face.xml'
file_path = r'/home/mohsen/Pictures'
face_name = 'mohsen'	

path = os.path.join(file_path, face_name) 
if not os.path.isdir(path): 
	os.mkdir(path) 

(width, height) = (130, 100)	 


face_cascade = cv2.CascadeClassifier(haar_file) 
webcam = cv2.VideoCapture(0) 

count = 1
while True:
	(_, im) = webcam.read() 
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
	faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
	for (x, y, w, h) in faces: 
		cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
		face = gray[y:y + h, x:x + w] 
		face_resize = cv2.resize(face, (width, height)) 
		cv2.imwrite('% s/% s.png' % (path, count), face_resize) 
	count += 1
	
	cv2.imshow('OpenCV', im) 
	key = cv2.waitKey(10) 

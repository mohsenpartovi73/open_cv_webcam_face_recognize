# Creating database 
# It captures images and stores them in datasets 
# folder under the folder name of sub_data 
import cv2, sys, numpy, os 
import cv2.face
import numpy as np
import cv2 as cv2
import cv2 as cv

haar_cascade = cv2.CascadeClassifier('/home/mohsen/Documents/trainee/tutorial/open_cv/face_recognition/haar_face.xml')
people = ['mohsen']
images = []
labels = []
path1 = r'/home/mohsen/Pictures'

# haar_cascade = cv2.CascadeClassifier('/home/mohsen/Documents/trainee/tutorial/open_cv/face_recognition/haar_face.xml')
def create_train():
    for person in people:
        path2 = os.path.join(path1,person)
        label = people.index(person)

        for img in os.listdir(path2):
            img_path = os.path.join(path2,img)

            img_array = cv2.imread(img_path)
            gray = cv.cvtColor(img_array,cv2.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            
            for(x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h,x:x+w]
                images.append(faces_roi)
                labels.append(label)

create_train()
# print(f'lenghth of images = {len(images)}')
# print(f'lenghth of lbels = {len(labels)}')
images = np.array(images,dtype='object')
labels = np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer.create()

face_recognizer.train(images,labels)

face_recognizer.save('face_train.yal')
np.save('images.npy',images)
np.save('labels.npy',labels)


face_recognizer = cv2.face.LBPHFaceRecognizer.create()
face_recognizer.read('face_train.yal')




# All the faces data will be 
# present this folder 
datasets = r'/home/mohsen/Pictures/mohsen'


# These are sub data sets of folder, 
# for my faces I've used my name you can 
# change the label here 
sub_data = 'mohsen'	

path = os.path.join(datasets, sub_data) 
if not os.path.isdir(path): 
	os.mkdir(path) 

# defining the size of images 
(width, height) = (130, 100)	 

#'0' is used for my webcam, 
# if you've any other camera 
# attached use '1' like this 
# face_cascade = cv2.CascadeClassifier(haar_cascade) 
face_cascade = haar_cascade
webcam = cv2.VideoCapture(0) 

# The program loops until it has 30 images of the face. 
count = 1
while count < 30: 
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
	if key == 27: 
		break

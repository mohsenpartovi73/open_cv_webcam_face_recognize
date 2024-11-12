import numpy as np
import cv2 as cv2
import cv2 as cv

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
haar_cascade = cv2.CascadeClassifier('/home/mohsen/Documents/trainee/tutorial/open_cv/face_recognition/haar_face.xml')
# images = np.load('images.npy')
# labels = np.load('labels.npy')

face_recognizer = cv2.face.LBPHFaceRecognizer.create()
face_recognizer.read('face_train.yal')

img = cv2.imread(r'/home/mohsen/Pictures/haar/valid/jerry_seinfeld/3.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)

faces_rect = haar_cascade.detectMultiScale(gray,1.1,8)
for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+h]

    label,confidence = face_recognizer.predict(faces_roi)
    print(f'label = {label} with confidence of {confidence} ')

    cv2.putText(img,str(people[label]),(20,20),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv2.imshow('detect',img)

cv2.waitKey()
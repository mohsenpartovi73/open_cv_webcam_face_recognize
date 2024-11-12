import os 
# import cv2.face
import cv2.face
import numpy as np
import cv2 as cv2

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
images = []
labels = []
path1 = r'/home/mohsen/Pictures/haar/train'

haar_cascade = cv2.CascadeClassifier('/home/mohsen/Documents/trainee/tutorial/open_cv/face_recognition/haar_face.xml')
def create_train():
    for person in people:
        path2 = os.path.join(path1,person)
        label = people.index(person)

        for img in os.listdir(path2):
            img_path = os.path.join(path2,img)

            img_array = cv2.imread(img_path)
            gray = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)

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



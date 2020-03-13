import cv2
import numpy as numpy
video=cv2.VideoCapture(0+cv2.CAP_DSHOW)
face_cascade=cv2.CascadeClassifier("C:\\Users\\Dell\\Desktop\\Python\\haarcascade_frontalface_default.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

while True:
    check,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for x,y,w,h in face:
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        id_,conf=recognizer.predict(roi_gray)
        if conf>=45 and conf<=85:
            print(id_)
        img_item="my_image.png"
        cv2.imwrite(img_item,roi_gray)
    cv2.imshow('frame',frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break;
video.release()
cv2.destroyAllWindows()
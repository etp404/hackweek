import cv2
import numpy

img = cv2.imread('resources/many_faces.jpg')
cv2.imshow('img', img)

face_cascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

height, width, depth = img.shape
faceMap = numpy.zeros((height, width,1), numpy.uint8)
for (x,y,w,h) in faces:
    cv2.rectangle(faceMap,(x,y),(x+w,y+h),255,-1)

cv2.waitKey(2000)
cv2.imshow('facemap', faceMap)
cv2.waitKey(0)
cv2.destroyAllWindows()
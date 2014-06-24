import cv2
import numpy

face_cascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")

img = cv2.imread('resources/tennis.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

height, width, depth = img.shape
faceMap = numpy.zeros((height, width, 1), numpy.float32)

for (x,y,w,h) in faces:
    centre = [(x+w)/2, y+h,2];
    cv2.rectangle(faceMap,(x,y),(x+w,y+h),1,-1)

gaussianWidth = max(width, height)
if gaussianWidth % 2 == 0 : # width of gaussian must be odd
    gaussianWidth+=1

ind = 0;
blurredFacemap = faceMap;
while ind<100:
    blurredFacemap = cv2.GaussianBlur(blurredFacemap, (101, 101), 20)
    ind+=1

blurredFacemap /= numpy.max(blurredFacemap)

print(numpy.min(blurredFacemap))
print(numpy.max(blurredFacemap))
#
# cv2.imshow('facemap', faceMap)
# cv2.waitKey(2000)

cv2.imshow('blurredfacemap', blurredFacemap)

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(blurredFacemap[y,x])

cv2.setMouseCallback('blurredfacemap',draw_circle)
cv2.waitKey(0)
cv2.destroyAllWindows()

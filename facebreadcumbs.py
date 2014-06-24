import cv2
import numpy

face_cascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
img = cv2.imread('resources/tennis.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

def get_intensity(event,x,y,flags,image):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x)
        print(y)
        print(image[y,x])
        print("---")

def create_breadcrumb_map(faces, height, width):
    breadcrumby, breadcrumbx = numpy.mgrid[0:height, 0:width]
    breadcrumb = numpy.zeros((height, width), numpy.float)
    for (x,y,w,h) in faces:
        x_centre = x+w/2
        y_centre = y+h/2
        breadcrumb -= numpy.sqrt(((x_centre-breadcrumbx)**2 + (y_centre-breadcrumby)**2).astype(float))
    breadcrumb = breadcrumb - numpy.min(breadcrumb)
    breadcrumb /= numpy.max(breadcrumb)
    return breadcrumb

height, width, depth = img.shape
breadcrumb = create_breadcrumb_map(faces, height, width)

cv2.imshow('breadcrumb', breadcrumb)
cv2.setMouseCallback('breadcrumb',get_intensity, breadcrumb)
cv2.waitKey(0)
cv2.destroyAllWindows()

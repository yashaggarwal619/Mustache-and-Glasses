import numpy as np
import cv2

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized

face_cascade        = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eyes_cascade        = cv2.CascadeClassifier('third-party/frontalEyes35x16.xml')
nose_cascade        = cv2.CascadeClassifier('third-party/Nose18x15.xml')
glasses             = cv2.imread("glasses.png", -1)
mustache            = cv2.imread('mustache.png',-1)
tyrion=cv2.imread('tyrion.png',-1)
cv2.imshow('d',tyrion)
frame=tyrion.copy()
gray     = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
face   =face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
frame=cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)
for (x,y,w,h) in face:
	roi_gray =gray[y:y+h ,x:x+h]
	roi_color=frame[y:y+h,x:x+h]
		#cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),3)
	eyes = eyes_cascade.detectMultiScale(roi_color, scaleFactor=1.5, minNeighbors=5)
	for (ex,ey,ew,eh) in eyes:
			#cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),4)
		roi_eyes=roi_gray[ey:ey+eh,ex:ex+ew]
		glasses2=image_resize(glasses.copy(),width=ew)
		gw,gh,gc=glasses2.shape
		for i in range(0,gw):
			for j in range(0,gh):
				if glasses2[i,j][3]!=0:
					roi_color[ey+i,ex+j]=glasses2[i,j]


		nose = nose_cascade.detectMultiScale(roi_color, scaleFactor=1.5, minNeighbors=5)
		for (nx,ny,nw,nh) in nose:
			#cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),3)
			roi_nose=roi_gray[ny:ny+nh,nx:nx+nw]
			mustache2=image_resize(mustache.copy(),width=nw+10)
			mw,mh,mc=mustache2.shape
			for i in range(0,mw):
				for j in range(0,mh):
					if mustache2[i,j][3]!=0:
						roi_color[ny+int(nh/2.0)+i+8,nx+j]=mustache2[i,j]

frame=cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)

cv2.imshow('image',frame)
cv2.imwrite('After.jpg',frame)
cv2.waitKey(0)		
cv2.destroyAllWindows()
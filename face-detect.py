import cv2 as cv

#Read the input image 
img = cv.imread('NCT-2.jpg')

#Load the cascade 
face_model = cv.CascadeClassifier('face-detect-model.xml')

#Convert to gray 
gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Detect faces
faces = face_model.detectMultiScale(gray_scale)

# Draw rectangle around the faces
for (x,y,w,h) in faces:
    cv.rectangle(img, (x,y),(x+w,y+h),(255,255,0), 2)

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
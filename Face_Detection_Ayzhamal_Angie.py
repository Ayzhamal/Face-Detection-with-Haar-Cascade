# Student:    Angie Ayzhamal
# Project 2:  Face Detection with Haar cascade classifier
# Date:       May 2, 2020

import cv2 as cv
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascade_eye.xml")
file = open("image_names.txt","r")
cvImg = []
faceImg = []

print("Reading images...")
for line in file:
    name = 'P2E_S5_C1.1/'+line.strip()
    img = cv.imread(name)
    cvImg.append(img)
    cv.imshow('img',img)
print("Finished reading images!")
file.close()
print("Detecting faces on Images...")
for img in cvImg:
    cv.imshow('img',img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(138,43,226),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv.imshow('img',img)
    faceImg.append(img)
print("Face Detection Done!")
delay = 1000
time = 0

height, width, layers = faceImg[0].shape
size = (width, height)
# to write an .avi video file with newly created images with face boxes - 30 frames per second
out = cv.VideoWriter('project_Angie_25fps.avi',cv.VideoWriter_fourcc(*'DIVX'), 25, size)

for i in range(3):
    for img in faceImg: #plays the images with the detected faces
        print("Displaying processed Images...")
        time += 100
        cv.imshow('img',img)
        while time < delay:
            time += 100
        cv.waitKey(25)
        time = 0
        out.write(img)
out.release()
print("Finished displaying processed Images!")
cv.waitKey(0)
cv.destroyAllWindows()

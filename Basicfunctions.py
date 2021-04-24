import cv2
import numpy as np
path = '/home/wargun/Pictures/spiderman2.jpg'
img = cv2.imread(path)


kernel = np.ones((5,5),np.uint8)
imGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imBlur = cv2.GaussianBlur(imGrey,(11,11),0)
imCanny = cv2.Canny(img, 150, 200)
imDialation = cv2.dilate(imCanny, kernel, iterations=1)
imEroded = cv2.erode(imDialation, kernel, iterations=1)
imResize = cv2.resize(img,(250,250))
imCropped = img[0:200,200:500]

cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)
cv2.rectangle(img,(0,0),(250,350),(0,0,255),2)
cv2.circle(img,(400,50),30,(255,255,0),5)
cv2.putText(img,"Hey Warlord",(300,150),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)

cv2.imshow('Image', img)
cv2.imshow('Gray image', imGrey)
cv2.imshow('Blur image', imBlur)
cv2.imshow('Canny image', imCanny)
cv2.imshow('Dilation image', imDialation)
cv2.imshow('Eroded image', imEroded)
cv2.imshow('Image', img)
cv2.imshow('Resize image', imResize)
cv2.imshow('Cropped image', imCropped)
cv2.waitKey(0)

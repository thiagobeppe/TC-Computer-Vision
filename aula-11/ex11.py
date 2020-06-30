import cv2
import numpy as np


# Fonte -> Documentação Open CV
#Q1
img = cv2.imread('ex8.jpg',0)
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Detect the values on image
cv2.imshow("Otsu Threshold", th3)
cv2.waitKey(0)


#Q2
img2 = cv2.imread('ex8.jpg',0)
img2 = cv2.medianBlur(img2,5)

img2Processada = cv2.Canny(img2, 50, 200)
cv2.imshow('Canny', img2Processada)
cv2.waitKey(0)
cv2.destroyAllWindows()
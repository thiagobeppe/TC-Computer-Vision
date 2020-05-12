import cv2
import numpy as np

#Reads the file
imgOriginal = cv2.imread("ex8.jpg", 0)
#Trying to reforce the board - Choose the median and laplacian filter to detect the board
imgTratada = cv2.medianBlur(imgOriginal,7)
imgFiltrada = cv2.Laplacian(imgTratada, cv2.CV_8U)
imgRealcada = cv2.subtract(imgOriginal, imgFiltrada)

#Applying the adapative binarization because the low luminosity
metodo = cv2.THRESH_BINARY_INV
imgBinarizada = cv2.adaptiveThreshold(imgRealcada, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,metodo, 11, 5)


# Confs for morphologic methods
typesMorph = [cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS, cv2.MORPH_RECT]
elementoEstruturante = cv2.getStructuringElement(typesMorph[0], (4,4))

#Erosão
imgProcessadaErode = cv2.erode(imgBinarizada, elementoEstruturante, iterations=1)


# #Dilatação
imgProcessadaDilate = cv2.dilate(imgBinarizada, elementoEstruturante, iterations=1)


#Abertura
imgProcessadaOpen = cv2.morphologyEx(imgBinarizada, cv2.MORPH_OPEN, elementoEstruturante)

#Fechamento
imgProcessadaClose = cv2.morphologyEx(imgBinarizada, cv2.MORPH_CLOSE, elementoEstruturante)


# Detect the values on image
imgFinal = cv2.dilate(cv2.morphologyEx(imgBinarizada, cv2.MORPH_CLOSE, elementoEstruturante), elementoEstruturante, iterations=1)
cv2.imshow("Final Close + Dilate", imgFinal)
cv2.waitKey(0)

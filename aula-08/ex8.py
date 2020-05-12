import cv2
import numpy as np

imgOriginal = cv2.imread("ex8.jpg", 0)
print(imgOriginal)
metodo = cv2.THRESH_BINARY_INV
ret, imgBinarizada = cv2.threshold(imgOriginal, 238, 255, metodo)

#Erosão
elementoEstruturante = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
imgProcessadaErode = cv2.erode(imgBinarizada, elementoEstruturante, iterations=1)
# cv2.imshow("Imagem Binarizada", imgBinarizada)
# cv2.imshow('Erosao', imgProcessadaErode)
# cv2.waitKey(0)

# #Dilatação
imgProcessadaDilate = cv2.dilate(imgBinarizada, elementoEstruturante, iterations=1)
# cv2.imshow("Imagem Binarizada", imgBinarizada)
# cv2.imshow('Dilatacao', imgProcessadaDilate)
# cv2.waitKey(0)

# #Abertura
imgProcessadaOpen = cv2.morphologyEx(imgBinarizada, cv2.MORPH_OPEN, elementoEstruturante)
# cv2.imshow("Imagem Binarizada", imgBinarizada)
# cv2.imshow('Abertura', imgProcessadaOpen)
# cv2.waitKey(0)

# #Fechamento
imgProcessadaClose = cv2.morphologyEx(imgBinarizada, cv2.MORPH_CLOSE, elementoEstruturante)
# cv2.imshow("Imagem Binarizada", imgBinarizada)
# cv2.imshow('Fechamento', imgProcessadaClose)
# cv2.waitKey(0)

#Tentativa
elementoEstruturante = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
imgProcessadaErode = cv2.erode(imgBinarizada, elementoEstruturante, iterations=1)
imgProcessadaCloseAndErode = cv2.morphologyEx(imgProcessadaErode, cv2.MORPH_CLOSE, elementoEstruturante)
imgProcessadaCloseAndErodeAndOpen = cv2.morphologyEx(imgProcessadaCloseAndErode, cv2.MORPH_OPEN, elementoEstruturante)
cv2.imshow("Imagem Erodida", imgBinarizada)
cv2.imshow('Processada Erosao + Fechamento', imgProcessadaCloseAndErode)
cv2.imshow('Processada Erosao + Fechamento + Abertura', imgProcessadaCloseAndErodeAndOpen)
cv2.waitKey(0)
# cv2.destroyAllWindows()
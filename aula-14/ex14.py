import cv2
import numpy as np

img = cv2.imread("fingerprint.png",0)

dX = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize = 1)
dY = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize = 1)


def calculateOrientations(gradientX,gradientY):
    orientation = np.ndarray(shape=(gradientX.shape[0],gradientX.shape[1]), dtype=float, order='F')
    for i in range (0 , gradientX.shape[0]):
        for j in range (0 , gradientX.shape[1]):
            vX = gradientX[i][j]
            vY = gradientY[i][j]
            result = cv2.fastAtan2(vX, vY)
            orientation[i][j] =result
    return orientation;

ang = calculateOrientations(dX,dY)
cv2.imshow("Original", img)
cv2.imshow("Sobel X", dX)
cv2.imshow("Sobel Y", dY)
cv2.imshow("Sobel Ang", ang)
cv2.waitKey(0)
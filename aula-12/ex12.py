import cv2
import numpy as np

#Q1
img = np.zeros((300,300))
img[150,150] = 255
img = np.uint8(img)

#Q2
img2 = np.zeros((300,300))
img2[150,125] = 255
img2[150,175] = 255
img2 = np.uint8(img2)

#Q3
img3 = np.zeros((300,300))
img3[110,125] = 255
img3[130,150] = 255
img3[150,175] = 255
img3 = np.uint8(img3)

#Q4
img4 = np.zeros((300,300))
img4[125,125] = 255
img4[125,175] = 255
img4[175,125] = 255
img4[175,175] = 255
img4 = np.uint8(img4)

imgs = [[img,0],[img2,1],[img3,2],[img4,1]]


def solution(transform, theta):
    lines = cv2.HoughLines(transform,1,np.pi/180,theta)
    transform = cv2.cvtColor(transform, cv2.THRESH_BINARY_INV)

    if lines is not None:
        for i in range(len(lines)):
            for rho,theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(transform,(x1,y1),(x2,y2),(0,0,255),2)

        cv2.imshow('Imagem', transform)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


for i in imgs:
    solution(i[0],i[1])
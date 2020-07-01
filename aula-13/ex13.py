import cv2
import numpy as np

img = cv2.imread('aeroporto.png')
cvtGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cvtGray,50,200,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,200)

if lines is not None:
    for i in range(len(lines)):
        for rho,theta in lines[i]:
            if (theta > -1*np.pi/180) and (theta < 1*np.pi/180):
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow('Imagem', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
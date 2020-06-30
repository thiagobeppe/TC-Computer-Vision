import cv2
import numpy as np


#Q1
image = np.ones((400,400))

color = (0,0,0)
r = 100
center = (200,200)
thickness = -1

structuringElement = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5,5)
)

image = cv2.circle(image, center, r, color, thickness)


cv2.imshow('Original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Q2
# https://iq.opengenus.org/connected-component-labeling/ <- Fonte
def connected_component_label(path="ex8-processed.jpg"):
    
    # Getting the input image
    img = cv2.imread(path, 0)
    # Converting those pixels with values 1-127 to 0 and others to 1
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    # Applying cv2.connectedComponents() 
    num_labels, labels = cv2.connectedComponents(img)
    
    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    
    
    # Showing Original Image
    cv2.imshow("Original",cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Showing Image after Component Labeling
    cv2.imshow("Labeled",cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

connected_component_label()
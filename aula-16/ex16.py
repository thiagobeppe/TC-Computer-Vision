import cv2
import numpy as np

img = cv2.imread("moedasw.jpg", 0)
blurred = cv2.GaussianBlur(img, (17, 17), 0)
outline = cv2.Canny(blurred, 30, 150)
cv2.imshow("The edges", outline)
cv2.waitKey(0)

#Componentes conexas
num_comp, labels = cv2.connectedComponents(outline)
label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue==0] = 0

#Take the outline
contornos, hierarquia = cv2.findContours(
    outline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

areas = []
for i in contornos:
    areas.append(cv2.contourArea(i))

areas = list(filter(lambda x: x > 30000, areas))
areas.sort()
values = ["1c","2c","5c","10c", "20c","50c","1E","2E"]
result = []
for i in range (0, len(areas)):
    result.append([areas[i], values[i]])
print(result)
cv2.imshow('ConectCounter', labeled_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
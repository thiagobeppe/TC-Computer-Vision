import cv2

img = cv2.imread('textura.png',0)

structuringElement = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5,5)
)

clsd = cv2.morphologyEx(
    img, cv2.MORPH_CLOSE, structuringElement, iterations=8
)


opnd = cv2.morphologyEx(
    clsd, cv2.MORPH_OPEN, structuringElement, iterations=20
)

grad = cv2.morphologyEx(
    opnd, cv2.MORPH_GRADIENT, structuringElement
)

final = img - grad
final = cv2.subtract(img, grad)

cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.imshow('Closed', clsd)
cv2.waitKey(0)
cv2.imshow('Opening', opnd)
cv2.waitKey(0)
cv2.imshow('Gradiente', grad)
cv2.waitKey(0)
cv2.imshow('Final', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
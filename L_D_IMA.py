import cv2
#import matplotlib.pyplot as plt

HP = cv2.imread('15M_P1_1.jpg')
H, W, _ = HP.shape
HP_R = cv2.resize(HP, (700,400))


cv2.imshow(' ',HP_R)

cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
def transition(I, J, alpha, beta):
    return (alpha*I+beta*J).astype(np.uint8);
I = cv2.imread('C:\\Users\\Hamed Khashehchi\\Desktop\\cv-lab2\\damavand.jpg');
J = cv2.imread('C:\\Users\\Hamed Khashehchi\\Desktop\\cv-lab2\\eram.jpg');
step=1;
start=0;
stop=250;
distant=stop-start;
smooth=2;
r=range(start,stop,step);
for m in r:
    K=transition(I,J,(stop-m)/distant,m/distant);
    cv2.imshow('windos',K);
    cv2.waitKey(step*smooth);
cv2.waitKey();
cv2.destroyAllWindows();
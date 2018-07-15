import numpy as np
import cv2
cap = cv2.VideoCapture('C:\\Users\\Hamed Khashehchi\\Desktop\\cv-lab3\\eggs.avi')

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')


out = cv2.VideoWriter('C:\\Users\\Hamed Khashehchi\\Desktop\\cv-lab3\\eggs-reverse.avi',fourcc, 30.0, (w,h))#chara 30?
i=[]
while True:
    ret, I = cap.read()
    if ret == False:
        break
    i.append(I);
i.reverse();
mat = np.array(i);
kk=0
print(mat.shape)
print(mat.size)
x,y,z,c=mat.shape;
while(kk<x):
    out.write(mat[kk])
    kk=kk+1;

cap.release()
out.release()

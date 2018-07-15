import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
def arbitrary(Image, color):
    temp=Image.ravel();
    histogram=[];
    for i in np.arange(256):
        histogram.append(0);
    for i in temp:
        histogram[i]=histogram[i]+1;
    cumulative=[];
    m=0
    for i in histogram:
        if m==0:
            cumulative.append(i)
        else:
            cumulative.append(cumulative[m-1]+i)
        m=m+1
    print(cumulative[m-1])
    a=0
    low=False
    count=0
    b=0
    big=False
    for i in cumulative:
        if np.float(i/cumulative[m-1])>0.05 and low==False:
            a=count;
            low=True;
        if(np.float(i/cumulative[m-1])>0.98):
            b=count;
            break;
        count+=1;
    a=Image.min();
    b=Image.max();
    print(a)
    print(b)
    print(histogram)
    print(cumulative)
    return (a,b);
def function(filename,a,b):
    I = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    f, axes = plt.subplots(2, 3)

    axes[0, 0].imshow(I, 'gray', vmin=0, vmax=255)
    axes[0, 0].axis('off')

    axes[1, 0].hist(I.ravel(), 256, [0, 256]);
    J = (I - a) * 255.0 / (b - a)
    J[J < 0] = 0
    J[J > 750] = 255
    J[J>700]=254
    J[J>650]=253
    J[J>600]=252
    J[J>550]=251
    J[J>500]=250
    J[J>450]=249
    J[J>400]=248
    J[J>350]=247
    J[J>300]=246
    J[J==255]=240
    J = J.astype(np.uint8)
    axes[0, 1].imshow(J, 'gray', vmin=0, vmax=255)
    axes[0, 1].axis('off')

    axes[1, 1].hist(J.ravel(), 256, [0, 256]);
    K = cv2.equalizeHist(I)
    axes[0, 2].imshow(K, 'gray', vmin=0, vmax=255)
    axes[0, 2].axis('off')

    axes[1, 2].hist(K.ravel(), 256, [0, 256]);
    plt.show()

fname1 = 'C:\\Users\\Hamed Khashehchi\\Desktop\\cv-lab3\\crayfish.jpg'
fname2 = 'C:\\Users\\Hamed Khashehchi\\Desktop\\cv-lab3\\map.jpg'
fname3 = 'C:\\Users\\Hamed Khashehchi\\Desktop\\cv-lab3\\train.jpg'
fname4 = 'C:\\Users\\Hamed Khashehchi\\Desktop\\cv-lab3\\branches.jpg'
fname5 = 'C:\\Users\\Hamed Khashehchi\\Desktop\\cv-lab3\\terrain.jpg'
'''
function(fname1,99,165);
cv2.waitKey()
function(fname2,157,212);
cv2.waitKey()
function(fname3,67,232);
cv2.waitKey()
function(fname4,143,223);
cv2.waitKey()
function(fname5,140,225);
'''

function(fname1,99,165);
cv2.waitKey()
(a,b)=arbitrary(cv2.imread(fname1, cv2.IMREAD_GRAYSCALE), (0, 256))
function(fname1,a,b)
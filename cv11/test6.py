import cv2
import numpy as np

I = cv2.imread('C:\\Users\\Hamed Khashehchi\\Desktop\\cv-lab11\\karimi.jpg',0)

# centre of the image
x=I.shape[1]/2.0
y=I.shape[0]/2.0

for theta in range(0,360):
    th = theta * np.pi / 180 # convert to radians

    R = np.array([[np.cos(th),-np.sin(th)],
                  [np.sin(th), np.cos(th)]])
    c = np.array([[x * (1 -np.cos(th))+np.sin(th)*y], [y * (1 -np.cos(th))-np.sin(th)*x]])
    #I'm so cool lol
    #mishe ye kare dge ham kard ke hamon c adi biai paeen age manzor one
    #GG WP
    t = np.zeros((2,1)) # you need to change this!
    t = np.add(t, c);
    # concatenate R and t to create the 2x3 transformation matrix
    M = np.hstack([R,t])

    J = cv2.warpAffine(I,M, (I.shape[1], I.shape[0]) )

    cv2.imshow('J',J)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


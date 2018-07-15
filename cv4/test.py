import numpy as np
import cv2

I = cv2.imread('C:\\Users\\Hamed Khashehchi\\Desktop\\cv-lab4\\isfahan.jpg', cv2.IMREAD_GRAYSCALE);
I = I.astype(np.float) / 255

sigma = 0.04  # initial standard deviation of noise
N = np.random.randn(*I.shape) * sigma
while True:
    N = np.random.randn(*I.shape) * sigma
    J = I+N;  # change this line so J is the noisy image

    cv2.imshow('snow noise', J)

    # press any key to exit
    key = cv2.waitKey(33)
    if key & 0xFF == ord('u'):
        sigma*=2
    elif key & 0xFF == ord('d'):
        sigma/=2
    elif key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('i'):
        print(sigma)

cv2.destroyAllWindows()

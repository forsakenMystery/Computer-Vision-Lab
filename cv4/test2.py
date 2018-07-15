import numpy as np
import cv2
from numpy.core.multiarray import dtype

I = cv2.imread('C:\\Users\\Hamed Khashehchi\\Desktop\\cv-lab4\\isfahan.jpg').astype(np.float64) / 255;

noise_sigma = 0.04

m = 1;

filter = 'b'

while True:
    N = np.random.randn(*I.shape) * noise_sigma

    if filter == 'b':
        F = np.ones((m, m), np.float64) / (m * m)

    elif filter == 'g':
        F=cv2.getGaussianKernel(m,-1);

    J = I + N;

    K = cv2.filter2D(J, -1, F);

    cv2.imshow('img', K)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('b'):
        filter = 'b'  # box filter
        print
        'Box filter'

    elif key == ord('g'):
        filter = 'g'  # filter with a Gaussian filter
        print
        'Gaussian filter'

    elif key == ord('+'):
        m = m + 2

    elif key == ord('-'):
        if m >= 3:
            m = m - 2

    elif key == ord('u'):
        noise_sigma*=2;

    elif key == ord('d'):
        noise_sigma/=2;

    elif key == ord('q'):
        break
    elif key == ord('i'):
        print('m= ', m)
        print('noise sigma= ',noise_sigma)
        print('kernel sigma= ',0.3*((m-1)*0.5 - 1) + 0.8)
        print('box filter= ',filter=='b')
        print('gausian filter= ',filter=='g')
        print('=====================================')
        print()
cv2.destroyAllWindows()

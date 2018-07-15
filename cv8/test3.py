import cv2
import numpy as np

I = cv2.imread('C:\\Users\\Hamed Khashehchi\\Desktop\\cv-lab8\\polygons.jpg')
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

ret, T = cv2.threshold(G, 220, 255, cv2.THRESH_BINARY_INV)
window_size = 5
soble_kernel_size = 3  # kernel size for gradients
alpha = 0.04
n=0
nc1, CC1 = cv2.connectedComponents(T)
for k in range(1, nc1):
    Ck = np.zeros(T.shape, dtype=np.float32)
    Ck[CC1 == k] = 1;
    Ck = cv2.GaussianBlur(Ck, (5, 5), 0)
    # Now, apply corner detection on Ck
    Ck = np.float32(Ck)
    H = cv2.cornerHarris(Ck, window_size, soble_kernel_size, alpha)
    H = H / H.max()
    C = np.uint8(H > 0.01) * 255
    n, CC, stats, centroids = cv2.connectedComponentsWithStats(C)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(G, np.float32(centroids), (5, 5), (-1, -1), criteria)
    J = Ck.copy()
    for i in range(1, n):
        cv2.circle(J, (int(corners[i, 0]), int(corners[i, 1])), 3, (255, 0, 255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(J, 'There are %d vertices!' % (n-1), (20, 30), font, 1, (255, 0, 255), 1)

    cv2.imshow('corners', J)
    cv2.waitKey(0)  # press any key




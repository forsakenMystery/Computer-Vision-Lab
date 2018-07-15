import numpy as np
import cv2


def point(good_match, key_point, side=0):
    points = good_match
    if side is 0:
        points = [key_point[m.queryIdx].pt for m in good_match]
    elif side is 1:
        points = [key_point[m.trainIdx].pt for m in good_match]
    points = np.array(points, dtype=np.float32)
    return points


def matches(desc1, desc2, alpha):
    matched = cv2.BFMatcher().knnMatch(desc1, desc2, 2)
    good_matches = [m1 for m1, m2 in matched if m1.distance < alpha * m2.distance]
    return good_matches


def detection(image):
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    key_points, desc = cv2.xfeatures2d.SIFT_create().detectAndCompute(g, None);
    return key_points, desc


def stitch(images, alpha):
    (img1, img2) = images
    (key1, desc1) = detection(img1)
    (key2, desc2) = detection(img2)
    good = matches(desc1, desc2, alpha)
    points1 = point(good, key1)
    points2 = point(good, key2, 1)
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    # imgs = np.zeros((max(img2.shape[0],img2.shape[0]), img1.shape[1]+img2.shape[1], 3))
    # print(img1.shape)
    # print(img1)
    # imgs[0:img1.shape[0], 0:img1.shape[1], 0] = img1[:,: ,0 ]
    I = cv2.drawMatches(img1, key1, img2, key2, good, None)
    # cv2.imshow('sift_keypoints1', I)
    # cv2.waitKey()
    # ii = 0
    # for p in points1:
    #     print(np.linalg.norm(points2[ii]-p))
    #     ii=ii+1
    # print(h)
    # ballance = np.array([[-2, 0, img2.shape[1]], [0, -2, 0], [0, 0, -2]])
    # print(-h-ballance)
    imgs = cv2.warpPerspective(img2, h, (img2.shape[1] + img1.shape[1], img2.shape[0]))
    # imgs = cv2.warpPerspective(img2, h, (img2.shape[1] + img1.shape[1], img2.shape[0]))
    imgs[0:img1.shape[0], 0:img1.shape[1]] = img1
    iii = 0
    # print(imgs.shape[0])
    for iii in reversed(range(0,imgs.shape[1])):
        if imgs[0, iii, 0] != 0:
            break
    # print(iii)
    return imgs[:, 0:iii, :]


def paranorma(image_paths, alpha=0.8, resize=True, size=(500, 500)):
    image_left = cv2.imread(image_paths[0])
    if resize:
        image_left = cv2.resize(image_left, size)
    for i in range(1, len(image_paths)):
        image_right = cv2.imread(image_paths[i])
        if resize:
            image_right = cv2.resize(image_right, size)
        image_left = stitch([image_left, image_right], alpha)
        # cv2.imshow('It is a trap', image_left)
        # cv2.waitKey()
    return image_left


# I1 = cv2.imread('KNTU-1.JPG')
# I2 = cv2.imread('KNTU-2.JPG')
# I3 = cv2.imread('KNTU-3.JPG')
# I4 = cv2.imread('KNTU-4.JPG')
# I1 = cv2.resize(I1, (500, 500))
# I2 = cv2.resize(I2, (500, 500))
# I3 = cv2.resize(I3, (500, 500))
# I4 = cv2.resize(I4, (500, 500))
# I = stitch([I1, I2], 0.7)
# cv2.imshow('tidies trap', I)
# J = stitch([I3, I4], 0.7)
# cv2.imshow('boobies trap', J)
# K = stitch([I, J], 0.7)
# cv2.imshow('tits trap', K)
# cv2.waitKey()
# M = stitch([I, I3], 0.7)
# cv2.imshow('asses trap', M)
# cv2.waitKey()
# cv2.destroyAllWindows()

# I1 = cv2.imread('Zabol-1.JPG')
# I2 = cv2.imread('Zabol-2.JPG')
# I3 = cv2.imread('Zabol-3.JPG')
# I4 = cv2.imread('Zabol-4.JPG')
# I1 = cv2.resize(I1, (500, 500))
# I2 = cv2.resize(I2, (500, 500))
# I3 = cv2.resize(I3, (500, 500))
# I4 = cv2.resize(I4, (500, 500))
# I = stitch([I1, I2], 0.7)
# cv2.imshow('tidies trap', I)
# J = stitch([I3, I4], 0.7)
# cv2.imshow('boobies trap', J)
# K = stitch([I, J], 0.7)
# cv2.imshow('tits trap', K)
# cv2.waitKey()
# M = stitch([I, I3], 0.7)
# cv2.imshow('asses trap', M)
# cv2.waitKey()
# cv2.destroyAllWindows()


KNTU = ['KNTU-1.JPG', 'KNTU-2.JPG', 'KNTU-3.JPG', 'KNTU-4.JPG']
Zabol = ['Zabol-1.JPG', 'Zabol-2.JPG', 'Zabol-3.JPG', 'Zabol-4.JPG']
image_kntu = paranorma(KNTU, 0.7)
image_zabol = paranorma(Zabol, 0.7)

cv2.imshow('trap', image_kntu)
cv2.waitKey()

cv2.imshow('trap', image_zabol)
cv2.waitKey()
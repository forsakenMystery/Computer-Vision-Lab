import numpy as np
import itertools
import cv2 as cv
import os
from skimage import data
from sklearn.svm import *
import pickle
import random


# global_variables
eps = 1e-7
personWidth = 64
personHeight = 128
projectDirectory = 'E:/Code/Python/Computer Vision/Image Processing/Pedastrion'
dataSetDirectory = projectDirectory+'/INRIAPerson'
trainDataSetDirectory = dataSetDirectory+'/train_64x128_H96'
positiveTrainDataSetDirectory = trainDataSetDirectory+'/pos'
negativeTrainDataSetDirectory = trainDataSetDirectory+'/neg'
testDataSetDirectory = dataSetDirectory+'/test_64x128_H96'
positiveTestDataSetDirectory = testDataSetDirectory+'/pos'
negativeTestDataSetDirectory = testDataSetDirectory+'/neg'
file_save_model = projectDirectory+'/Calculated.pkl'


def go_loading(i, total, show):
    percentage = i*100/total
    p = percentage
    s = "["
    k = 0
    while percentage > 0:
        s += "*"
        percentage -= 2
        k += 1
    k_prime = k
    while k < 50:
        k += 1
        s += " "
    if not(show == k_prime):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(s + "]" + str("{:2.3f}".format(p)).zfill(6) + "%  -->  " + str(i).zfill(4) + "/" + str(total).zfill(4))
    return k_prime


def loading_train_data_set(train, x, y, positive=True):
    files = os.listdir(train)
    i = 0
    total = len(files)
    show = -1
    hog = cv.HOGDescriptor(projectDirectory+'\\toxicity.xml')
    if positive:
        print("start loading "+str(total)+" positive instances")
        for f in files:
            image = data.imread(positiveTrainDataSetDirectory+"\\"+f, as_gray=True)
            image = image[16:16 + personHeight, 16:16 + personWidth]
            destination = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

            x.append(hog.compute(destination))
            y.append(1)
            show = go_loading(i, total, show)
            i = i + 1
    else:
        print("start loading " + str(total) + " negative instances")
        for f in files:
            image = data.imread(negativeTrainDataSetDirectory+"\\"+f, as_gray=True)
            destination = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            for SS in range(10):
                row = random.randint(0, np.shape(destination)[0] - 128)
                col = random.randint(0, np.shape(destination)[1] - 64)
                destination = destination[row:row+personHeight, col:col+personWidth]
                x.append(hog.compute(destination))
                y.append(0)
            show = go_loading(i, total, show)
            i = i + 1


def loading_test_data_set(test, x, y, positive=True):
    files = os.listdir(test)
    i = 0
    total = len(files)
    show = -1
    hog = cv.HOGDescriptor(projectDirectory+'\\toxicity.xml')
    if positive:
        print("start loading "+str(total)+" positive instances")
        for f in files:
            image = data.imread(positiveTestDataSetDirectory+"\\"+f, as_gray=True)
            image = image[3:3 + personHeight, 3:3 + personWidth]
            destination = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            x.append(hog.compute(destination))
            y.append(1)
            show = go_loading(i, total, show)
            i = i + 1
    else:
        print("start loading " + str(total) + " negative instances")
        for f in files:
            image = data.imread(negativeTestDataSetDirectory+"\\"+f, as_gray=True)
            destination = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            for SS in range(10):
                row = random.randint(0, np.shape(destination)[0] - 128)
                col = random.randint(0, np.shape(destination)[1] - 64)
                destination = destination[row:row + personHeight, col:col + personWidth]
                x.append(hog.compute(destination))
                y.append(0)
            show = go_loading(i, total, show)
            i = i + 1


def create_train_data_set():
    X = []
    Y = []
    print("loading train data set:\n")
    loading_train_data_set(positiveTrainDataSetDirectory, X, Y)
    print('\n=================================\n=================================\n')
    loading_train_data_set(negativeTrainDataSetDirectory, X, Y, False)
    print('\n=================================\n=================================\n=================================\n')
    return np.array(X).reshape((np.array(X).shape[0], np.array(X).shape[1])), np.array(Y)


def create_test_data_set():
    X = []
    Y = []
    print("loading test data set:\n")
    loading_test_data_set(positiveTestDataSetDirectory, X, Y)
    print(np.array(X).shape)
    print('\n=================================\n=================================\n')
    loading_test_data_set(negativeTestDataSetDirectory, X, Y, False)
    print(np.array(X).shape)
    print('\n=================================\n=================================\n=================================\n')
    return np.array(X).reshape((np.array(X).shape[0], np.array(X).shape[1])), np.array(Y)


def non_max_suppression(rectangle, minimum_threshold=0.01):
    if len(rectangle) == 0:
        return []
    pickle_rick = []
    x1 = rectangle[:, 0]
    y1 = rectangle[:, 1]
    x2 = rectangle[:, 2]
    y2 = rectangle[:, 3]
    area = np.abs(np.multiply(np.add(np.subtract(x2, x1), 1), np.add(np.subtract(y2, y1), 1)))
    print(area)
    index = np.argsort(y2)
    print(index)
    while len(index) > 0:
        length = len(index) - 1
        i = index[length]
        pickle_rick.append(i)
        max_x1 = np.maximum(x1[i], x1[index[:length]])
        max_y1 = np.maximum(y1[i], y1[index[:length]])
        max_x2 = np.minimum(x2[i], x2[index[:length]])
        max_y2 = np.minimum(y2[i], y2[index[:length]])
        width = np.maximum(0, np.abs(max_x2 - max_x1 + 1))
        height = np.maximum(0, np.abs(max_y2 - max_y1 + 1))
        intersection_over_union = (width * height) / area[index[:length]]
        index = np.delete(index, np.concatenate(([length], np.where(intersection_over_union > minimum_threshold)[0])))
    return rectangle[pickle_rick].astype("int")


def main(save=False, load=False):
    if load:
        with open(file_save_model, 'rb') as file:
            model = pickle.load(file)
        # X, Y = create_train_data_set()
        # XX, YY = create_test_data_set()
        # print("train accuracy:")
        # print(str("{:2.3f}".format(model.score(X, Y) * 100)) + "%")
        # print("test accuracy:")
        # print(str("{:2.3f}".format(model.score(XX, YY) * 100)) + "%")
        # print("done")
        supportvectors = []
        supportvectors.append(np.dot(model.dual_coef_, model.support_vectors_)[0])
        print(np.array(supportvectors, dtype=np.float64).shape)
        supportvectors.append([model.intercept_])
        supportvectors = list(itertools.chain(*supportvectors))
        hog_load = cv.HOGDescriptor(projectDirectory + '\\toxicity.xml')
        to = np.array(supportvectors, dtype=np.float64)
        print(to.shape)
        hog_load.setSVMDetector(to)
        return hog_load
    else:
        X, Y = create_train_data_set()
        print("Train data loaded:")
        print("shape of x:")
        print(X.shape)
        print("shape of y:")
        print(Y.shape)
        XX, YY = create_test_data_set()
        print("Test data loaded:")
        print("shape of x:")
        print(XX.shape)
        print("shape of y:")
        print(YY.shape)
        model = SVC(kernel='linear', gamma='auto', C=0.01, max_iter=-1, tol=1e-4, coef0=1)
        print('fitting started')
        model.fit(X, Y)
        print('fitting ended')
        print("train accuracy:")
        print(str("{:2.3f}".format(model.score(X, Y) * 100)) + "%")
        print("test accuracy:")
        print(str("{:2.3f}".format(model.score(XX, YY)*100))+"%")
        print("done")
        if save:
            with open(file_save_model, 'wb') as file:
                pickle.dump(model, file)
        supportvectors= []
        supportvectors.append(np.dot(model.dual_coef_, model.support_vectors_)[0])
        print(np.array(supportvectors, dtype=np.float64).shape)
        supportvectors.append([model.intercept_])
        supportvectors = list(itertools.chain(*supportvectors))
        hog_save = cv.HOGDescriptor(projectDirectory + '\\toxicity.xml')
        to = np.array(supportvectors, dtype=np.float64)
        print(to.shape)
        hog_save.setSVMDetector(to)
        return hog_save


if __name__ == '__main__':
    HOG = main(load=True)

    image = cv.imread('E:\\Code\\Python\\Computer Vision\\Image Processing\\Pedastrion\\INRIAPerson\\Test\\pos\\person_216.png')
    destination = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    destination = cv.normalize(destination, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    rectangles, weights = HOG.detectMultiScale(destination, winStride=(8, 8), scale=1.01, padding=(0, 0), finalThreshold=70, useMeanshiftGrouping=0)
    print(rectangles)
    # rectangles = non_max_suppression(rectangle=rectangles)
    # print(rectangles)
    a=0
    print(weights.shape)
    print(weights[a])
    for (x, y, w, h) in rectangles:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(image, str(weights[a]), (x+5, int(y+h/2+random.randint(-10, 10))), cv.FONT_HERSHEY_SIMPLEX, 0.44, (random.randint(0, 255), random.randint(120, 255), random.randint(200, 255)), 2)
        a+=1
    cv.imshow("Detections", image)
    cv.waitKey()
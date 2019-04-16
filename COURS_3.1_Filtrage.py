import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


def niveaudegris(nomimage):
    img = cv2.imread(nomimage, 0)
    return img


def nbpixelsgris(nomimage, couleur='gris'):
    if couleur == 'gris':
        img = cv2.imread(nomimage)
    return img.shape


def luminance(img):
    print('Luminance :')
    print('\n Min : ')
    print(np.min(img))
    print('\n Max : ')
    print(np.max(img))
    print('\n Mean : ')
    print(np.mean(img))
    print('\n Ecart-type: ')
    print(np.std(img))


def histogramme(img):
    plt.hist(img.ravel(), 256, [0, 256]);
    plt.show()


def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(img)
    return gray


def egaliserhistogramme(img):
    imgegalisee = cv2.equalizeHist(img)
    plt.hist(imgegalisee.ravel(), 256, [0, 256]);
    plt.show()
    cv2.imshow('image egalisee', imgegalisee)
    return img


def enregistrer(img, nom):
    cv2.imwrite((nom + '.jpg'), img)


def filtremoyenneur(img, nb):
    kernel = np.ones((nb, nb), np.float32) / (nb * nb)
    img = cv2.filter2D(img, -1, kernel)
    cv2.imshow('filtre moyenneur', img)
    return img


def filtregaussien(img, nb):
    img = cv2.GaussianBlur(img, (nb, nb), 0)
    cv2.imshow('filtre gaussien', img)
    return img


def filtremedian(img, nb):
    img = cv2.medianBlur(img, nb)
    cv2.imshow('filtre médiane', img)
    return img


def imagebinarise(img, t, type=cv2.THRESH_BINARY):
    # type = cv2.THRESH_BINARY_INV ou cv2.THRESH_TRUNC ou cv2.THRESH_TOZERO ou cv2.THRESH_TOZERO_INV
    ret, thresh1 = cv2.threshold(img, t, 255, type)
    cv2.imshow('Binarise a la valeur:', thresh1)
    return thresh1


def adaptivethresholdgaussian(img):
    # type = cv2.THRESH_BINARY_INV ou cv2.THRESH_TRUNC ou cv2.THRESH_TOZERO ou cv2.THRESH_TOZERO_INV
    thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('Adaptive Binaire Gaussian', thresh1)
    return thresh1

def cb_seuillage(x):
    ret, binary = cv2.threshold(img, cv2.getTrackbarPos("SeuilT", "Binary"), 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Binary", binary)

def trackbarmouvement(imgrgb):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) --> Pour trackbar à rajouter dans le main()
    cv2.namedWindow("Binary", 0)
    cv2.createTrackbar("SeuilT", "Binary", 100, 255, cb_seuillage)

if __name__ == "__main__":
    img = cv2.imread('voilier_oies_blanches.jpg')
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def niveaudegris(nomimage):
    img = cv2.imread(nomimage, 0)
    cv2.imshow('image', img)
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
    cv2.imwrite((nom+'.jpg'), img)

def filtremoyenneur(img, nb):
    kernel = np.ones((nb, nb), np.float32) / (nb*nb)
    img = cv2.filter2D(img, -1, kernel)
    cv2.imshow('filtre moyenneur', img)
    return img

def filtregaussien(img, nb):
    img = cv2.GaussianBlur(img,(nb,nb),0)
    cv2.imshow('filtre gaussien', img)
    return img

def filtremedian(img, nb):
    img = cv2.medianBlur(img, nb)
    cv2.imshow('filtre médiane', img)
    return img


if __name__ == "__main__":
    img = niveaudegris('Image_epave.jpg')
    filtremedian(img, 3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def enregistrer(nom, img):
    cv2.imwrite((nom+'.jpg'), img)

if __name__ == "__main__":
    img = niveaudegris('voilier_oies_blanches.jpg')
    #img = egaliserhistogramme(img)
    #enregistrer('egaliser', img)
    egaliserhistogramme(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


def niveaudegris2(nomimage):
    img = cv2.imread(nomimage)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', gray)


def niveaudegris(nomimage):
    img = cv2.imread(nomimage, 0)
    cv2.imshow('image', img)
    return img


def nbpixelsgris(nomimage, couleur='gris'):
    if couleur == 'gris':
        img = cv2.imread(nomimage)
    return img.shape


if __name__ == "__main__":
    img = niveaudegris('voilier_oies_blanches.jpg')
    print(nbpixelsgris('voilier_oies_blanches.jpg'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

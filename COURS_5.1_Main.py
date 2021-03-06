import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


def niveaudegris(nomimage):
    img = cv2.imread(nomimage, 0)
    cv2.imshow('Niveau de Gris :', img)
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


def filtremoyenneur(img, nb=5):
    kernel = np.ones((nb, nb), np.float32) / (nb * nb)
    img = cv2.filter2D(img, -1, kernel)
    cv2.imshow('filtre moyenneur', img)
    return img


def filtregaussien(img, nb=5):
    img = cv2.GaussianBlur(img, (nb, nb), 0)
    cv2.imshow('filtre gaussien', img)
    return img


def filtremedian(img, nb=5):
    img = cv2.medianBlur(img, nb)
    cv2.imshow('filtre médiane', img)
    return img

def reductionbruit(img, filtre="null"):
    if filtre == "gaussien":
        img = filtregaussien(img)
    elif filtre == "median":
        img = filtremedian(img)
    else:
        img = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imshow('reduction de bruit', img)
    return img

def imagebinarisee(img, t=200, type=cv2.THRESH_BINARY):
    # type = cv2.THRESH_BINARY_INV ou cv2.THRESH_TRUNC ou cv2.THRESH_TOZERO ou cv2.THRESH_TOZERO_INV
    ret, thresh1 = cv2.threshold(img, t, 255, type)
    cv2.imshow('Binarise a la valeur:', thresh1)
    return thresh1

def imagebinariseeflottant(img, seuil = 127):
    imgbinarisee = img > seuil
    return imgbinarisee

def imagebinariseeinv(img, t=200, type=cv2.THRESH_BINARY_INV):
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

def erode(img):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    cv2.imshow('Erode', erosion)

def dilate(img):
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    cv2.imshow('Dilation', dilation)

def openingclosing(img):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Opening-Closing :',closing)

def morphological(img):
    kernel = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow('Gradient:', gradient)

def etiquageconnexes(img):
    _, comp_conn = cv2.connectedComponents(img)
    plt.figure()
    plt.imshow(comp_conn), plt.title('composantes connexes')
    plt.colorbar()
    plt.show()


def detectcontourssobel(img):
    #gradientXY
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    #SOBEL en X =     [1 0 -1,
                     # 2 0 -2,
                     # 1 0 -1]
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobelxy = np.abs(sobelx) + np.abs(sobely)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 5), plt.imshow(sobelxy, cmap='gray')
    plt.title('Sobel X et Y'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 6), plt.imshow(imagebinariseeflottant(sobelxy), cmap='gray')
    plt.title('Sobel X et Y - Binarisé'), plt.xticks([]), plt.yticks([])
    plt.show()
    return sobelxy

def detectcontourscanny(img):
    edges = cv2.Canny(img, 100, 200, apertureSize=3, L2gradient=1)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    img = niveaudegris('img_ds.jpg')
    detectcontourscanny(img)
    img = reductionbruit(img)
    img = detectcontourscanny(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

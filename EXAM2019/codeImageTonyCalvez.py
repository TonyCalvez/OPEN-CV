# -*- coding: utf-8 -*-
"""
TONY CALVEZ - FIPA20
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np

def show_webcam(mirror=False):
    count = 1
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        
        if cv2.waitKey(1) == 99:
            print("Capture")
            cv2.imwrite("src%d.jpg" % count, img)
            count = count+1
    cv2.destroyAllWindows()

    
def saveImage(img, name):
    cv2.imwrite(name + '.jpg', img)

def niveaudegris(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def filtregaussien(img, nb=5):
    img = cv2.GaussianBlur(img, (nb, nb), 0)
    return img

def detectcontourscanny(img):
    edges = cv2.Canny(img,100,200)
    return edges

def egaliserhistogramme(img):
    imgegalisee = cv2.equalizeHist(img)
    return imgegalisee

def imagebinarisee(img, t=4, type=cv2.THRESH_BINARY):
    ret, thresh1 = cv2.threshold(img, t, 255, type)
    return thresh1
    
def openingclosing(img):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing

def findcountours(im):
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(im,contours,-1,(255,0,0),3)

plt.show()
if __name__ == '__main__':
    #LES CAPTURES SE FONT AVEC LA LETTRE C
    #show_webcam()
    
    plt.subplot(4, 4, 1)
    img1 = cv2.imread('src1.jpg')
    plt.imshow(img1, cmap='gray')
    plt.title('img1')
    
    plt.subplot(4, 4, 2)
    img2 = cv2.imread('src2.jpg')
    plt.imshow(img2, cmap='gray')
    plt.title('img2')
    
    
    #QUESTION 2
    plt.subplot(4, 4, 3)
    img1 = cv2.imread('src1.jpg')
    img2 = cv2.imread('src2.jpg')
    diff = cv2.absdiff(img1, img2)
    plt.imshow(diff, cmap='gray')
    plt.title('diff')
    #NOUS N'OBSERVONS AUCUNE MODIFICATION ENTRE LES DEUX IMAGES, NOUS OBTENONS DONC DU NOIR
    
    
    #QUESTION 3
    plt.subplot(4, 4, 4)
    img1 = cv2.imread('src1.jpg')
    img3 = cv2.imread('src3.jpg')
    diff13C = cv2.absdiff(img3, img1)
    plt.imshow(diff13C, cmap='gray')
    plt.title('diff13C')
    saveImage(diff13C, 'diff13C')
    #NOUS POUVONS ISOLER LES ELEMENTS QUI ONT ETE MODIFIES ENTRE DEUX FRAMES
    #NOUS POUVONS DONC OBSERVER AVANT L'EMPLACEMENT DU CUBE, LE FOND DE L'IMAGE MASQUE PAR LE CUBE.
    
    #QUESTION 4 - NIVEAU DE GRIS
    src1NG = niveaudegris(img1)
    saveImage(src1NG, 'src1NG')
    src3NG = niveaudegris(img3)
    saveImage(src3NG, 'src3NG')
    
    #QUESTION 5 - GAUSSIEN
    plt.subplot(4, 4, 5)
    src3NG_gauss10 = filtregaussien(src3NG, nb=3)
    plt.imshow(src3NG_gauss10, cmap='gray')
    plt.title('src3NG_gauss segma = 1.0')
    
    plt.subplot(4, 4, 6)
    src3NG_gauss25 = filtregaussien(src3NG, nb=9)
    plt.imshow(src3NG_gauss25, cmap='gray')
    plt.title('src3NG_gauss segma = 2.5')
    
    #QUESTION 5 - ABSOLUE GAUSSIEN
    plt.subplot(4, 4, 7)
    diff13NG = cv2.absdiff(src3NG_gauss10, src3NG_gauss25)
    plt.imshow(diff13NG, cmap='gray')
    plt.title('diff13NG')
    saveImage(diff13C, 'diff13NG')
    #NOUS OBSERVONS BEAUCOUP PLUS LES CONTOURS, CAR LE FILTRE GAUSSIEN VA FILTRER ET REALISER UNE MEILLEURE DEMARQUATION.
    
    #QUESTION 6 - DETECTION DE CONTOURS AVEC CANNY
    plt.subplot(4, 4, 8)
    diff13NG_contours = detectcontourscanny(egaliserhistogramme(filtregaussien(diff13NG)))
    plt.imshow(diff13NG_contours, cmap='gray')
    plt.title('diff13NG - Contours')
    saveImage(diff13NG_contours, 'diff13NG_contours')
    #LE RESULTAT N'EST PAS TRES PERTINENT J'AI DONC EGALISE L'HISTOGRAMME POUR AVOIR UN MEILLEUR CONTRASTE
    
    #QUESTION 8 - BINARISATION AVEC SEUIL
    plt.subplot(4, 4, 9)
    plt.hist(diff13NG.ravel(), 256, [0, 256]);
    plt.title('diff13NG_Histo')
    
    plt.subplot(4, 4, 10)
    bin13NG = imagebinarisee(diff13NG, t=4)
    plt.imshow(bin13NG, cmap='gray')
    plt.title('bin13NG')
    saveImage(bin13NG, 'bin13NG')
    
    plt.subplot(4, 4, 11)
    plt.hist(diff13C.ravel(), 256, [0, 256]);
    plt.title('diff13NG_Histo')
    
    plt.subplot(4, 4, 12)
    bin13C = imagebinarisee(diff13C, t=20)
    plt.imshow(bin13C, cmap='gray')
    plt.title('bin13C')
    saveImage(bin13C, 'bin13C')
    #LE RESULTAT EST PARFAIT AVEC BIN13C -> DENOMBREMENT SUPER FACILE
    #LES SEUILS SONT FIXES FACILEMENT AVEC L'HISTOGRAMME -> GENERE
    bin13 = bin13C
    saveImage(bin13, 'bin13')
    
    #QUESTION 9 - FERMETURE ET OUVERTURE
    plt.subplot(4, 4, 13)
    bin13post = openingclosing(bin13)
    plt.imshow(bin13post, cmap='gray')
    plt.title('bin13POST')
    saveImage(bin13post, 'bin13post')
    #LE RESULTAT EST BIEN PLUS PROPRE SUR LES BORDS DE L'OBJET ET CECI VA EVITER D'AVOIR DES POINTS QUI POURRAIT GENERER DES OBJETS SUPLLEMENTAIRES DANS UN DENOMBREMENT.
    
    #QUESTION 10 - INTERPRETATION
    plt.subplot(4, 4, 14)
    bin13post_contours = findcountours(bin13post)
    plt.imshow(bin13post_contours, cmap='gray')
    plt.title('bin13post_contours')
    saveImage(bin13post_contours, 'composantes_connexes')
    #MALHEURESEMENT J'AI PAS PU AVOIR DE VISUALISATION DES CONTOURS
    #TOUTEFOIS LES COORDONNEES PEUVENT FACILEMENT SUIVRE ET FAIRE DU TRACKING SUR L'IMAGE. LA RECONNAISSANCE FACIALE EN CHINE EST A LA POINTE DE CETTE TECHNOLOGIE.
    
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
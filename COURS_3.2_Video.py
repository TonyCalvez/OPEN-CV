import cv2

def main():

    def cb_seuillage(x):
        ret, binary = cv2.threshold(thresh1,cv2.getTrackbarPos("SeuilT","Binary"),255,cv2.THRESH_BINARY_INV)
        cv2.imshow("Binary",binary)

    img = cv2.imread('voilier_oies_blanches.jpg')
    thresh1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    seuil = 100
    cv2.namedWindow("Binary",0)
    cv2.createTrackbar("SeuilT","Binary",seuil,255,cb_seuillage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#3b img_ds bruitee
# -> image binaire est tres bruitee
# -> 2 solutions pour ameliorer le resultat
#       -> a) prefiltrage de img_ds.jpg avant binaritation
#       -> b) post-traitement de l'image binaire a l'aide d'operateurs de morphologie mathematique (CF TE4)


if __name__ == "__main__":
    main()

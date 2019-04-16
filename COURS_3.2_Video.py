import cv2
import numpy as np

def main():

    cap = cv2.VideoCapture(1)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "_main_":
      main()

#3b img_ds bruitee
# -> image binaire est tres bruitee
# -> 2 solutions pour ameliorer le resultat
#       -> a) prefiltrage de img_ds.jpg avant binaritation
#       -> b) post-traitement de l'image binaire a l'aide d'operateurs de morphologie mathematique (CF TE4)

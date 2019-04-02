import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

if __name__ == "__main__":


    # Fx * 512 = 16 cycles par image selon x
    # Fy * 512 = 41 cycles par image selon y
    Fx = 0.03
    Fy = 0.08

    # Z(m, n) -> m = y = lignes et n = x = colonnes
    # I(m,n) = math.sin(2*math.pi*(Fx.n + Fy.m)) : sinusoide

    Z = np.fromfunction(lambda m, n: np.sin(2 * np.pi * (Fx * n + Fy * m)), (1024, 1024)) #augmente la définition

    tfd_Z = np.fft.fft2(Z)
    magnitude_spectrum = np.abs(np.fft.fftshift(tfd_Z))

    plt.figure()
    plt.subplot(121)
    plt.imshow(Z, cmap='gray')
    plt.title('Mire (image avant FFT)')
    plt.xlabel('n')
    plt.ylabel('m')

    plt.subplot(122)
    plt.imshow(np.log2(1 + magnitude_spectrum), origin='upper', extent=(-0.5, 0.5, 0.5, -0.5), cmap='jet')
    plt.title('Spectre Amplitudes Normalisé')

    plt.colorbar()
    plt.show() 

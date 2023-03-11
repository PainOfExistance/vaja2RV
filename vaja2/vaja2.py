﻿import cv2
import numpy as np


def my_roberts(slika):
    # Define the Roberts kernels
    rx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    ry = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    # Define the Roberts kernels
    irx = cv2.filter2D(slika, -1, rx)
    iry = cv2.filter2D(slika, -1, ry)
    # Compute the magnitude of the Roberts edge detection
    robertsSlika = np.sqrt(np.square(irx) + np.square(iry))
    robertsSlika = np.uint8(robertsSlika)

    lower_white = np.array([30], dtype=np.uint8)
    upper_white = np.array([255], dtype=np.uint8)

    mask = cv2.inRange(robertsSlika, lower_white, upper_white)
    dst = cv2.bitwise_and(robertsSlika, robertsSlika, mask=mask)

    slika_robov = np.zeros((slika.shape[0], slika.shape[1], 3), np.uint8)
    finslika = np.zeros((slika.shape[0], slika.shape[1], 3), np.uint8)

    for i in range(0, slika.shape[0]):
        for j in range(0, slika.shape[1]):
            slika_robov[i, j] = [0, 0, dst[i, j]]
            finslika[i, j] = [slika[i, j], slika[i, j], slika[i, j]]

    slika_robov = cv2.addWeighted(finslika, 0.4, slika_robov, 1.4, 3)

    return slika_robov

def my_prewitt(slika):
    # Define the Prewitt kernels
    kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Apply the Prewitt kernels to obtain the gradient images
    prewittx = cv2.filter2D(slika, -1, kx)
    prewitty = cv2.filter2D(slika, -1, ky)

    # Combine the gradient images using the np.sqrt function to get the final Prewitt edge detection image
    pewitt = prewittx + prewitty
    slika_robov = np.zeros((slika.shape[0], slika.shape[1], 3), np.uint8)
    finslika = np.zeros((slika.shape[0], slika.shape[1], 3), np.uint8)

    for i in range(0, slika.shape[0]):
        for j in range(0, slika.shape[1]):
            slika_robov[i, j] = [0, 0, pewitt[i, j]]
            finslika[i, j] = [slika[i, j], slika[i, j], slika[i, j]]

    slika_robov = cv2.addWeighted(finslika, 0.4, slika_robov, 1.4, 3)

    return slika_robov 

def my_sobel(slika):
    sobelx_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobely_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Apply the Sobel kernels to the image
    sobelx = cv2.filter2D(gray, -1, sobelx_kernel)
    sobely = cv2.filter2D(gray, -1, sobely_kernel)

    # Compute the magnitude of the gradient
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    cv2.imshow("Sobel edge detection", sobel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0
    return slika_robov 

"""
def canny(slika, sp_prag, zg_prag):
    return slika_robov 

"""

def spremeni_kontrast(slika, alfa, beta):
    for i in range(0, slika.shape[0]):
        for j in range(0, slika.shape[1]):
            temp = alfa * slika[i, j] + beta
            slika[i, j] = temp
    return slika

img = cv2.imread(
    "C:/Users/MatejPC/Desktop/Sola/2.letnik/4.semester/rv/vaja2/lenna.png", 0
)
img = cv2.resize(img, (800, 800))
alfa = float(input("vnestie alfa za kontrast:"))
beta = float(input("vnestie beta za kontrast:"))
img = spremeni_kontrast(img, alfa, beta)
choice = input("Vnestie tehnologijo 1.roberts 2.pewitt 3.sobel 4.cany")

if choice=="1":
    finimg = my_roberts(img)

elif choice=="2":
    finimg=my_prewitt(img)

elif choice=="3":
    print("todo sobel")


cv2.imshow("Edge Detection", finimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
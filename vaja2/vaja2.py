from gzip import READ
import cv2
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

    alfa = float(input("vnestie alfa za kontrast:"))
    beta = float(input("vnestie beta za kontrast:"))
    robertsSlika = spremeni_kontrast(robertsSlika, alfa, beta)

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
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Apply the Prewitt kernels to obtain the gradient images
    prewittx = cv2.filter2D(slika, -1, kernelx)
    prewitty = cv2.filter2D(slika, -1, kernely)

    # Combine the gradient images using the np.sqrt function to get the final Prewitt edge detection image
    slika_robov = prewittx + prewitty

    # Display the result
    cv2.imshow('Prewitt Edge Detection', slika_robov)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return slika_robov 

"""
def my_sobel(slika):
    return slika_robov 

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

my_prewitt(img)
finimg = my_roberts(img)

cv2.imshow("Edge Detection", finimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
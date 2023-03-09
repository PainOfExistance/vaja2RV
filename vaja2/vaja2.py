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
    slika_robov = np.sqrt(np.square(irx) + np.square(iry))
    slika_robov = np.uint8(slika_robov)

    alfa = float(input("vnestie alfa za kontrast:"))
    beta = float(input("vnestie beta za kontrast:"))
    slika_robov = spremeni_kontrast(slika_robov, alfa, beta)
    # Display the results
    cv2.imshow("Original Image", slika)
    cv2.imshow("Roberts Edge Detection", slika_robov)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return slika_robov


"""
def my_prewitt(slika):
    return slika_robov 

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


img = cv2.imread("C:/Users/MatejPC/Desktop/Sola/2.letnik/4.semester/rv/vaja2/vegeta.jpg", 0)
alfa = float(input("vnestie alfa za kontrast:"))
beta = float(input("vnestie beta za kontrast:"))
img = spremeni_kontrast(img, alfa, beta)


my_roberts(img)
"""
# Define the Roberts kernels
rx = np.array([[1, 0], [0, -1]], dtype=np.float32)
ry = np.array([[0, 1], [-1, 0]], dtype=np.float32)

# Apply the Roberts kernels to the image
irx = cv2.filter2D(img, -1, rx)
iry = cv2.filter2D(img, -1, ry)

# Compute the magnitude of the Roberts edge detection
img_roberts = np.sqrt(np.square(irx) + np.square(iry))
img_roberts = np.uint8(img_roberts)

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Roberts Edge Detection', img_roberts)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
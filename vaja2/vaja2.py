import cv2
import numpy as np


def my_roberts(slika):
    # Define the Roberts kernels
    rx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    ry = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    # Define the Roberts kernels
    irx = cv2.filter2D(img, -1, rx)
    iry = cv2.filter2D(img, -1, ry)
    # Compute the magnitude of the Roberts edge detection
    slika_robov = np.sqrt(np.square(irx) + np.square(iry))
    slika_robov = np.uint8(slika_robov)

    # Display the results
    cv2.imshow('Original Image', img)
    cv2.imshow('Roberts Edge Detection', slika_robov)
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

def spremeni_kontrast(slika, alfa, beta):
    return 0
"""

img = cv2.imread('C:/Users/MatejPC/Desktop/Sola/2.letnik/4.semester/rv/vaja2/vegeta.jpg', 0)
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
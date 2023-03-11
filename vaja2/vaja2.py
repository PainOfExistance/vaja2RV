import cv2
import numpy as np

def merge_images(slika, edgeSlika):
    slika_robov = np.zeros((slika.shape[0], slika.shape[1], 3), np.uint8)
    slika = cv2.cvtColor(slika, cv2.COLOR_GRAY2BGR)
    slika_robov[:, :, 2] = edgeSlika
    slika_robov = cv2.addWeighted(slika, 0.4, slika_robov, 1.4, 3)
    return slika_robov

def my_roberts(slika):
    rx = np.array([[1, 0], [0, -1]])
    ry = np.array([[0, 1], [-1, 0]])

    irx = cv2.filter2D(slika, -1, rx)
    iry = cv2.filter2D(slika, -1, ry)

    robertsSlika = np.sqrt(np.square(irx) + np.square(iry))
    robertsSlika = np.uint8(robertsSlika)

    robertsSlika = spremeni_kontrast(robertsSlika, 9, 0)

    lower_white = np.array([30], dtype=np.uint8)
    upper_white = np.array([255], dtype=np.uint8)

    mask = cv2.inRange(robertsSlika, lower_white, upper_white)
    filterRoberts = cv2.bitwise_and(robertsSlika, robertsSlika, mask=mask)

    return merge_images(slika, filterRoberts)


def my_prewitt(slika):
    kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    prewittx = cv2.filter2D(slika, -1, kx)
    prewitty = cv2.filter2D(slika, -1, ky)
    pewitt = np.abs(prewittx) + np.abs(prewitty)

    return merge_images(slika, pewitt)


def my_sobel(slika):
    sxk = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    syk = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    sobelx = cv2.filter2D(slika, -1, sxk)
    sobely = cv2.filter2D(slika, -1, syk)
    sobel = np.abs(sobelx) + np.abs(sobely)

    return merge_images(slika, sobel)


def canny(slika, sp_prag, zg_prag):
    canny = cv2.Canny(slika, sp_prag, zg_prag)
    return merge_images(slika, canny) 


def spremeni_kontrast(slika, alfa, beta):
    ^#slika=alfa * slika + beta
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
while True:
    choice = input("Vnestie tehnologijo 1.roberts 2.prewitt 3.sobel 4.cany")

    if choice == "1":
        finimg = my_roberts(img)

    elif choice == "2":
        finimg = my_prewitt(img)

    elif choice == "3":
        finimg = my_sobel(img)

    elif choice == "4":
        sp_prag = float(input("vnesite spodnji prag:"))
        zg_prag = float(input("vnestie zgornji prag:"))
        finimg = canny(img, sp_prag, zg_prag)

    cv2.imshow("Edge Detection", finimg)
    cv2.waitKey(0);
    cv2.destroyAllWindows();

    izb = input("Želite preiskusiti drug način iskanja robov? y/n")
    if izb=="n":
        break

import cv2
img1 = cv2.imread('example1.jpg')
img2 = cv2.imread('coded1.jpg')
psnr = cv2.PSNR(img1, img2)

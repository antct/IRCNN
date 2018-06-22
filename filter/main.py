import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

img_noise = cv2.imread('./image/B.png')
img_origin = cv2.imread('./image/B_ori.png')
b, g, r = cv2.split(img_noise)
img_noise = cv2.merge([r, g, b])

mask = []
for row in range(img_noise.shape[0]):
    line_mask = []
    for col in range(img_noise.shape[1]):
        if img_noise[row][col][0] != 0 or img_noise[row][col][1] != 0 or img_noise[row][col][2] != 0:
            line_mask.append(1)
        else:
            line_mask.append(0)
    mask.append(line_mask)


def IHMeanMask(image):
    img_mask = image.copy()
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if mask[row][col] == 1:
                img_mask[row][col] = img_noise[row][col]
            else:
                pass
    return img_mask

def IHMeanOperator(roi, q):
    roi = roi.astype(np.float64)
    return np.mean((roi)**(q+1))/np.mean((roi)**(q))


def IHMeanFilter(image, q):
    image_filter = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            image_filter[i-1, j-1] = IHMeanOperator(image[i-1:i+2, j-1:j+2], q)
    image_filter = (image_filter-np.min(image))*(255/np.max(image))
    return image_filter.astype(np.uint8)

def MSE(aisle, channel):
    s, n = 0.0, 0
    for row in range(img_origin.shape[0]):
        for col in range(img_origin.shape[1]):
            s += (img_origin[row][col][channel] - aisle[row][col]) ** 2
            n += 1
    return s/n


def IHMean(image, q):
    r, g, b = cv2.split(image)
    r = IHMeanFilter(r, q)
    g = IHMeanFilter(g, q)
    b = IHMeanFilter(b, q)
    print(MSE(r, 0), MSE(g, 1), MSE(b, 2))
    return cv2.merge([r, g, b])


def IHMeanEpoch(image):
    img_IHMean = IHMeanMask(image)
    img_IHMean = IHMean(img_IHMean, 0.1)
    return img_IHMean

plt.subplot(221), plt.imshow(img_noise)
img_IHMean = IHMean(img_noise, 0.1)
plt.subplot(222), plt.imshow(img_IHMean)
img_IHMean = IHMeanEpoch(img_IHMean)
plt.subplot(223), plt.imshow(img_IHMean)
for i in range(3):
    img_IHMean = IHMeanEpoch(img_IHMean)
plt.subplot(224), plt.imshow(img_IHMean)
plt.show()
r, g, b = cv2.split(img_IHMean)
img_IHMean = cv2.merge([b, g, r])
cv2.imwrite("1.png", img_IHMean)

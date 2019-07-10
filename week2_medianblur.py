# coding: utf-8

import cv2
import numpy as np


def img_gray(img):
    """
    转换成灰度图
    :param img:
    :return:
    """
    h = img.shape[0]
    w = img.shape[1]
    img1 = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            img1[i, j] = 0.144 * img[i, j, 0] + 0.587 * img[i, j, 1] + 0.299 * img[i, j, 2]
    return img1


def noise(img, snr):
    """
    添加椒盐噪声
    :param img:
    :param snr: 信噪比
    :return:
    """
    h = img.shape[0]
    w = img.shape[1]
    img1 = img.copy()
    # print(img1)
    # print("*"*10)
    # print(img)
    sp = h * w  # 计算图像像素点个数
    NP = int(sp * (1 - snr))  # 计算图像椒盐噪声点个数
    for i in range(NP):
        randx = np.random.randint(0, h)  # 生成一个 0 至 h 之间的随机整数
        randy = np.random.randint(0, w)  # 生成一个 0 至 w 之的间随机整数
        if np.random.random() <= 0.5:  # np.random.random()生成一个 0 至 1 之间的浮点数
            img1[randx, randy] = 0
        else:
            img1[randx, randy] = 255
    return img1


def median(image):
    """
    中值滤波
    :param image:
    :return:
    """
    # 边界0填充
    img = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    h = img.shape[0]
    w = img.shape[1]
    img1 = np.zeros((h, w), np.uint8)
    for i in range(1, h-1):
        for j in range(1, w-1):
            temporary = np.zeros(9, np.uint8)
            s = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    temporary[s] = img[i + k, j + l]
                    s += 1
            # 给滤波器窗口排序，把原滤波器中心值的数替换成排序的中间值
            for y in range(8):
                count = y
                for x in range(y, 8):
                    if temporary[count] > temporary[x + 1]:
                        count = x + 1
                temporary[y], temporary[count] = temporary[count], temporary[y]
            median = temporary[4]
            img1[i, j] = median
    return img1


image = cv2.imread('C:/Users/tang/Desktop/0.jpg')
# print(cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT))
# image = np.resize(image, (64, 128, 3))
img_gray = img_gray(image)
SNR = 0.95  # 将椒盐噪声信噪比设定为0.95
noiseimage = noise(img_gray, SNR)
medianimage = median(noiseimage)
cv2.imshow("grayimage", img_gray)
cv2.imshow("noiseimage", noiseimage)
cv2.imshow("medianimage", medianimage)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()



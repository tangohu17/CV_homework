
# coding: utf-8

# In[1]:


import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

class Homework_week1(object):
    def __init__(self):
        pass

    def img_read_gray(self):
        img_gray = cv2.imread('C:/Users/tang/Desktop/0.jpg', 0)
        cv2.imshow('lenna', img_gray)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()

    def img_spilt(self, y_start, y_end, x_start, x_end):
        img = cv2.imread('C:/Users/tang/Desktop/0.jpg')
        img_crop = img[y_start:y_end, x_start:x_end]
        cv2.imshow('img_crop', img_crop)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()

    def color_spilt(self):
        img = cv2.imread('C:/Users/tang/Desktop/0.jpg')
        B, G, R = cv2.split(img)
        cv2.imshow('B', B)
        cv2.imshow('G', G)
        cv2.imshow('R', R)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()

    def random_light_color(self):
        img = cv2.imread('C:/Users/tang/Desktop/0.jpg')
        # brightness
        B, G, R = cv2.split(img)
        b_rand = random.randint(-50, 50)
        if b_rand == 0:
            pass
        elif b_rand > 0:
            lim = 255 - b_rand
            B[B > lim] = 255
            B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
        elif b_rand < 0:
            lim = 0 - b_rand
            B[B < lim] = 0
            B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

        g_rand = random.randint(-50, 50)
        if g_rand == 0:
            pass
        elif g_rand > 0:
            lim = 255 - g_rand
            G[G > lim] = 255
            G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
        elif g_rand < 0:
            lim = 0 - g_rand
            G[G < lim] = 0
            G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

        r_rand = random.randint(-50, 50)
        if r_rand == 0:
            pass
        elif r_rand > 0:
            lim = 255 - r_rand
            R[R > lim] = 255
            R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
        elif r_rand < 0:
            lim = 0 - r_rand
            R[R < lim] = 0
            R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

        img_merge = cv2.merge((B, G, R))
        cv2.imshow('img_random_color', img_merge)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()

    def gamma_correction(self, gamma, x_rows, y_cols):
        img_dark = cv2.imread('C:/Users/tang/Desktop/0.jpg')
        # print(img_dark.dtype)
        invGamma = 1.0 / gamma
        table = []
        for i in range(256):
            table.append(((i / 255.0) ** invGamma) * 255)
        table = np.array(table).astype("uint8")
        table_LUT = cv2.LUT(img_dark, table)
        # return cv2.LUT(img_dark, table)
        cv2.imshow('img_dark', img_dark)
        cv2.imshow('img_brighter', table_LUT)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()

        # histogram
        img_small_brighter = cv2.resize(table_LUT, (int(table_LUT.shape[0] * x_rows), int(table_LUT.shape[1] * y_cols)))
        plt.hist(table_LUT.flatten(), 256, [0, 256], color='r')
        img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        cv2.imshow('Color input image', img_small_brighter)
        cv2.imshow('Histogram equalized', img_output)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()

    def rotation(self):
        img = cv2.imread('C:/Users/tang/Desktop/0.jpg')
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 1)  # center, angle, scale
        img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        cv2.imshow('rotated lenna', img_rotate)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()

    def similarity_transform(self, center_1, center_0, angle, scale):
        img = cv2.imread('C:/Users/tang/Desktop/0.jpg')
        M = cv2.getRotationMatrix2D((img.shape[1] / center_1, img.shape[0] / center_0), angle, scale)  # center, angle, scale
        img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        cv2.imshow('rotated lenna', img_rotate)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()

    def Affine_Transform(self):
        img = cv2.imread('C:/Users/tang/Desktop/0.jpg')
        rows, cols, ch = img.shape
        pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(img, M, (cols, rows))

        cv2.imshow('affine lenna', dst)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()

    def  perspective_transform(self):
        img = cv2.imread('C:/Users/tang/Desktop/0.jpg')
        height, width, channels = img.shape
        random_margin = 60
        x1 = random.randint(-random_margin, random_margin)
        y1 = random.randint(-random_margin, random_margin)
        x2 = random.randint(width - random_margin - 1, width - 1)
        y2 = random.randint(-random_margin, random_margin)
        x3 = random.randint(width - random_margin - 1, width - 1)
        y3 = random.randint(height - random_margin - 1, height - 1)
        x4 = random.randint(-random_margin, random_margin)
        y4 = random.randint(height - random_margin - 1, height - 1)

        dx1 = random.randint(-random_margin, random_margin)
        dy1 = random.randint(-random_margin, random_margin)
        dx2 = random.randint(width - random_margin - 1, width - 1)
        dy2 = random.randint(-random_margin, random_margin)
        dx3 = random.randint(width - random_margin - 1, width - 1)
        dy3 = random.randint(height - random_margin - 1, height - 1)
        dx4 = random.randint(-random_margin, random_margin)
        dy4 = random.randint(height - random_margin - 1, height - 1)

        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        M_warp = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, M_warp, (width, height))

        cv2.imshow('lenna_warp', img_warp)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()



if __name__ == '__main__':
    img_tansform = Homework_week1()
    print(img_tansform.img_read_gray())
    print(img_tansform.img_spilt(0, 200, 0, 200))
    print(img_tansform.color_spilt())
    print(img_tansform.random_light_color())
    print(img_tansform.gamma_correction(2, 0.5, 0.5))
    print(img_tansform.rotation())
    print(img_tansform.similarity_transform(2, 2, 45, 0.5))
    print(img_tansform.Affine_Transform())
    print(img_tansform.perspective_transform())



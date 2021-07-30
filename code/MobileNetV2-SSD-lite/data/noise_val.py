import os
import random
import glob
import numpy as np
import os
import cv2 as cv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--thresh', type=float, default=0.01)
opt = parser.parse_args()

thresh=opt.thresh

def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


#验证集
path ='./val_noise'
xmlfilepath =glob.glob( '{}/*.xml'.format(path))
jpgfilepath =glob.glob( '{}/*.jpg'.format(path))

xmlfilepath=np.sort(xmlfilepath)
jpgfilepath=np.sort(jpgfilepath)


num = len(jpgfilepath)
list = range(num)

fval = open('./val_noise.txt', 'w+')

for i in list:
    name = jpgfilepath[i][:-4] + '\n'
    fval.write(name)

fval.close()

for i in list:
    raw_img = cv.imread(jpgfilepath[i],cv.IMREAD_COLOR)
    img = raw_img.copy()
    noise = True
    if noise and random.random()<0.5:
        img = sp_noise(img, thresh)
    if i % 10==0:
        print("OK! {} images.".format(i))
    cv.imwrite('{}'.format(jpgfilepath[i]), img)



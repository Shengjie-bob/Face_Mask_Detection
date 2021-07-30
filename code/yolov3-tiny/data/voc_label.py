import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import  cv2 as cv
from PIL import Image


classes = ["face", "face_mask"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    in_file = open('%s.xml'%(image_id))
    out_file = open('%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()


    size = root.find('size')

    if root.find('size'):
        size = root.find('size')
        if int(size.find('width').text) == 0 or int(size.find('height').text) == 0:
            print('%s.jpg长或宽为0' % (image_id))
            img = Image.open('%s.jpg' % (image_id))
            w = int(img.width)
            h = int(img.height)
        else:
            w = int(size.find('width').text)
            h = int(size.find('height').text)
    else:
        print('%s.jpg没有大小信息' % (image_id))
        img = Image.open('%s.jpg' % (image_id))
        w = int(img.width)
        h = int(img.height)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

#训练集
image_ids = open('./train.txt').read().strip().split()
list_file = open('./train.txt', 'w')
for image_id in image_ids:
    list_file.write('%s.jpg\n'%(image_id))
    convert_annotation(image_id)
list_file.close()



#测试集
image_ids = open('./val.txt').read().strip().split()
list_file = open('./val.txt', 'w')
for image_id in image_ids:
    list_file.write('%s.jpg\n'%(image_id))
    convert_annotation(image_id)
list_file.close()
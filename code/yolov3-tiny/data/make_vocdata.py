import os
import random
import glob
import numpy as np
import os


trainval_percent = 0.1
train_percent = 0.9

#训练集
path ='./train'

xmlfilepath =glob.glob( '{}/*.xml'.format(path))
jpgfilepath =glob.glob( '{}/*.jpg'.format(path))

xmlfilepath=np.sort(xmlfilepath)
jpgfilepath=np.sort(jpgfilepath)



# #排列图片
# i =0
# for file in jpgfilepath:
#     i=i+1
#     os.rename(file, os.path.join(path, '%05d' % i + ".jpg"))
#
# #排列xml文件
# i =0
# for file in jpgfilepath:
#     i=i+1
#     os.rename(file, os.path.join(path, '%05d' % i + ".xml"))



num = len(jpgfilepath)
list = range(num)

ftrain = open('./train.txt', 'w+')

for i in list:
    name = jpgfilepath[i][:-4] + '\n'
    ftrain.write(name)
ftrain.close()



#验证集
path ='./val'
xmlfilepath =glob.glob( '{}/*.xml'.format(path))
jpgfilepath =glob.glob( '{}/*.jpg'.format(path))

xmlfilepath=np.sort(xmlfilepath)
jpgfilepath=np.sort(jpgfilepath)


num = len(jpgfilepath)
list = range(num)

fval = open('./val.txt', 'w+')

for i in list:
    name = jpgfilepath[i][:-4] + '\n'
    fval.write(name)

fval.close()


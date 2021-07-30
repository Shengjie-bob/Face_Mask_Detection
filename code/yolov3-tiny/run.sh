#! /bin/bash

#训练
# python train.py --argument --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 2
#测试训练好的模型
# python test.py --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/best.pt --batch-size 10 --device 2 --iou-thres 0.6 --iouv 0.6
#用训练好的模型测试sample中图片
# python detect.py --cfg cfg/yolov3-tiny-mask.cfg --names data/mask.names --weights weights/best_server.pt --source ./samples --output outputimg
#测试下载的模型
# python test.py --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/best_server.pt --batch-size 10 --device 2 --iou-thres 0.6 --iouv 0.6
#用下载的模型测试sample中图片
# python detect.py --cfg cfg/yolov3-tiny-mask.cfg --names data/mask.names --weights weights/best_server.pt --source ./samples --output outputimg



#超参数验证
# python train.py --argument --evolve --generate --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 2
#使用adam优化器结果和SGD方法对比
# python train.py --adam --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 2
# python train.py --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 2
#使用hsv空间变化结果的对比
# python train.py --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 2
# python train.py --argument --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 2
#使用soft_nms 和其超参数sigma
# python train.py --soft_nms --sigma 0.1 --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 3
# python train.py --soft_nms --sigma 0.5 --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 2
# python train.py --soft_nms --sigma 1 --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 2
# python train.py --soft_nms --sigma 3 --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 2
# python train.py --soft_nms --sigma 5 --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 2
# python train.py --soft_nms --sigma 7 --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 2
# python train.py --soft_nms --sigma 9 --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 2
#测试k-means的个数的区别
# python train.py --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask-kmeans6.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 3
# python train.py --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask-kmeans8.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 3
# python train.py --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask-kmeans4.cfg --data data/mask.data --weights weights/yolov3-tiny.conv.15  --device 3

#测试不同噪声下的检测效果 
#没有噪声
# python test.py --name noise0 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/best_server.pt --batch-size 10 --device 2 --iou-thres 0.6 --iouv 0.6
# cd data 
# python noise_val.py --thresh 0.01
# cd ..
# python test.py --name noise0.01 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask_noise.data --weights weights/best_server.pt --batch-size 10 --device 2 --iou-thres 0.6 --iouv 0.6

# cd data 
# python noise_val.py --thresh 0.03
# cd ..
# python test.py --name noise0.03 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask_noise.data --weights weights/best_server.pt --batch-size 10 --device 2 --iou-thres 0.6 --iouv 0.6

#在有噪声的情况下训练
# python train.py --name noise --noise --argument --epochs 5 --batch-size 10 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask_noise.data --weights weights/yolov3-tiny.conv.15  --device 2
#将训练好的模型在非noise数据测试
# python test.py --name noisemodel-0 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/best_noise.pt --batch-size 10 --device 2 --iou-thres 0.6 --iouv 0.6
#将训练好的模型在noise数据测试
# python test.py --name noisemodel-0.03 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask_noise.data --weights weights/best_noise.pt --batch-size 10 --device 2 --iou-thres 0.6 --iouv 0.6


#测试不同的图像分辨率下的结果
# python test.py --name resolution128 --img-size 128 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/best_server.pt --batch-size 10 --device 2 --iou-thres 0.6 --iouv 0.6
# python test.py --name resolution256 --img-size 256 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/best_server.pt --batch-size 10 --device 2 --iou-thres 0.6 --iouv 0.6
# python test.py --name resolution384 --img-size 384 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/best_server.pt --batch-size 10 --device 2 --iou-thres 0.6 --iouv 0.6
# python test.py --name resolution512 --img-size 512 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/best_server.pt --batch-size 10 --device 2 --iou-thres 0.6 --iouv 0.6
# python test.py --name resolution640 --img-size 640 --cfg cfg/yolov3-tiny-mask.cfg --data data/mask.data --weights weights/best_server.pt --batch-size 10 --device 2 --iou-thres 0.6 --iouv 0.6

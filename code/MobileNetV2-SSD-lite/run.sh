#! /bin/bash

#训练模型
# python train_ssd.py --argument --datasets data --validation_dataset data --net mb2-ssd-lite --pretrained_ssd models/mb2-ssd-lite-mp-0_686.pth --num_epochs 20 --batch_size 64 --debug_steps 10 --use_cuda True
#测试训练好的模型
# python eval_ssd.py --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-0-origin.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6
#测试训练好的模型在sample中图片的表现
# python run_ssd_example.py mb2-ssd-lite models/mb2-ssd-lite-Epoch-19-origin.pth models/voc-model-labels.txt ./sample ./output
#测试下载的模型
# python eval_ssd.py --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-best.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6
#测试下载的模型在sample中图片的表现
# python run_ssd_example.py mb2-ssd-lite models/mb2-ssd-lite-Epoch-19-best.pth models/voc-model-labels.txt ./sample ./output


#超参数
# python train_ssd.py --evolve --datasets data --validation_dataset data --net mb2-ssd-lite --pretrained_ssd models/mb2-ssd-lite-mp-0_686.pth --num_epochs 10 --batch_size 64 --debug_steps 10 --use_cuda True
#hard做测试
# python eval_ssd.py --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-Loss-2.070806778710464.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6
#使用soft做测试
# python eval_ssd.py --nms_method soft --sigma 0.1 --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-Loss-2.070806778710464.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6
# python eval_ssd.py --nms_method soft --sigma 0.3 --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-Loss-2.070806778710464.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6
# python eval_ssd.py --nms_method soft --sigma 0.5 --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-Loss-2.070806778710464.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6
# python eval_ssd.py --nms_method soft --sigma 1 --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-Loss-2.070806778710464.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6
# python eval_ssd.py --nms_method soft --sigma 3 --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-Loss-2.070806778710464.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6
# python eval_ssd.py --nms_method soft --sigma 5 --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-Loss-2.070806778710464.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6

#测试图片的预处理的作用
#不加
# python train_ssd.py --name noprecessing --datasets data --validation_dataset data --net mb2-ssd-lite --pretrained_ssd models/mb2-ssd-lite-mp-0_686.pth --num_epochs 20 --batch_size 64 --debug_steps 10 --use_cuda True
# python eval_ssd.py --name noprecessing --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-noprecessing.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6
#加
# python train_ssd.py --argument --datasets data --validation_dataset data --net mb2-ssd-lite --pretrained_ssd models/mb2-ssd-lite-mp-0_686.pth --num_epochs 20 --batch_size 64 --debug_steps 10 --use_cuda True

#测试不同优化器效果
# #adam
# python train_ssd.py --argument --adam --name adam-True --datasets data --validation_dataset data --net mb2-ssd-lite --pretrained_ssd models/mb2-ssd-lite-mp-0_686.pth --num_epochs 20 --batch_size 64 --debug_steps 10 --use_cuda True
# python eval_ssd.py --name adam --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-adam-True.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6
# #sgd
# python train_ssd.py --argument --name argu-True --datasets data --validation_dataset data --net mb2-ssd-lite --pretrained_ssd models/mb2-ssd-lite-mp-0_686.pth --num_epochs 20 --batch_size 64 --debug_steps 10 --use_cuda True
# python eval_ssd.py --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-argu-True.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6


#测试不同噪声幅度下的检测效果 
# cd data 
# python noise_val.py --thresh 0.01
# cd ..
# python eval_ssd.py --noise --name noise0.01 --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-Loss-2.070806778710464.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6

# cd data 
# python noise_val.py --thresh 0.03
# cd ..
# python eval_ssd.py --noise --name noise0.03 --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-Loss-2.070806778710464.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6

#训练不同噪声幅度下的检测效果 
# python train_ssd.py --name noise --noise --argument --datasets data --validation_dataset data --net mb2-ssd-lite --pretrained_ssd models/mb2-ssd-lite-mp-0_686.pth --num_epochs 20 --batch_size 64 --debug_steps 10 --use_cuda True
# python eval_ssd.py --noise --name noisemodel0.03 --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-noise.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6
# python eval_ssd.py --name noisemodel0 --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-noise.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6


#测试不同分辨率的结果  暂时未实现 （TODO）
# python eval_ssd.py --image_size 128 --name resolution128 --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-Loss-2.070806778710464.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6
# python eval_ssd.py --image_size 256 --name resolution256 --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-Loss-2.070806778710464.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6
# python eval_ssd.py --image_size 384 --name resolution384 --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-Loss-2.070806778710464.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6
# python eval_ssd.py --image_size 512 --name resolution512 --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-Loss-2.070806778710464.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6
# python eval_ssd.py --image_size 640 --name resolution640 --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-19-Loss-2.070806778710464.pth --dataset data --label_file models/voc-model-labels.txt --iou_threshold 0.6

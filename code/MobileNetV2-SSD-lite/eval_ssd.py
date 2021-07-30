import torch
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
from vision.ssd.data_preprocessing import PredictionTransform
import argparse
import pathlib
import numpy as np
import logging
import sys
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor


parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model",type=str)

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=False)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.6, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
parser.add_argument("--sigma", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--name", type=str, default="hard",help='save test{}.txt path name')
parser.add_argument("--noise", action='store_true',help='whether add noise to test or not')
parser.add_argument("--image_size", type=int, default=300, help="inference size (pixels).")


args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric,class_index):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        """
        加入 P R 计算 :排列scores
        """
        scores = scores[sorted_indexes]
        r_score = 0.1
        p_score = 0.9
        ""
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    r = np.interp(-r_score, -scores, recall)
    p = np.interp(-r_score,-scores,precision)

    np.save('./{}/recall_{}_{}.npy'.format(args.eval_dir,class_index,args.iou_threshold), recall)
    np.save('./{}/precision_{}_{}.npy'.format(args.eval_dir,class_index,args.iou_threshold), precision)

    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall) ,p ,r
    else:
        return measurements.compute_average_precision(precision, recall) , p, r


if __name__ == '__main__':
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    timer = Timer()

    class_names = [name.strip() for name in open(args.label_file).readlines()]

    #测试的transform 如果是原始的300则不进入
    # if args.image_size == 300 :
    #     pre_transform = None
    # else:
    #     pre_transform = None
    pre_transform = None
    
    if args.dataset_type == "voc":

        dataset = VOCDataset(args.dataset,transform=pre_transform,is_test=True,noise=args.noise)

    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)

    if args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)  

    timer.start("Load Model")
    net.load(args.trained_model)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')

    if args.net == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method,sigma=args.sigma, device=DEVICE)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    results = []
    for i in range(len(dataset)):
        print("process image", i)
        timer.start("Load Image")
        image = dataset.get_image(i)
        print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(image)
        print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))
    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )

    f = open("./{}/test_{}_{}.txt".format(args.eval_dir, args.iou_threshold,args.name), 'w+')
    aps = [];ps=[];rs=[];f1s=[]
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue

        if args.iou_threshold ==0.6:

            ious=np.linspace(0.5, 0.95, 10)
            maps =[];mp=[];mr=[];mf1=[]
            for iou_thre  in ious:
                prediction_path = eval_path / f"det_test_{class_name}.txt"
                ap, P, R = compute_average_precision_per_class(
                    true_case_stat[class_index],
                    all_gb_boxes[class_index],
                    all_difficult_cases[class_index],
                    prediction_path,
                    iou_thre,
                    args.use_2007_metric,
                    class_index
                )
                F1 = 2 * P * R / (P + R + 1e-16)
                maps.append(ap);mp.append(P);mr.append(R);mf1.append(F1)
            ap=np.mean(maps); P=np.mean(mp);R=np.mean(mr);F1=np.mean(mf1)

        else:
            prediction_path = eval_path / f"det_test_{class_name}.txt"

            ap ,P ,R= compute_average_precision_per_class(
                true_case_stat[class_index],
                all_gb_boxes[class_index],
                all_difficult_cases[class_index],
                prediction_path,
                args.iou_threshold,
                args.use_2007_metric,
                class_index
            )
            F1 = 2 * P * R / (P + R + 1e-16)
        aps.append(ap)
        ps.append(P)
        rs.append(R)
        f1s.append(F1)
        print(f"{class_name}: {ap} {P} {R} {F1}")
        f.write(f"{class_name}: {ap} {P} {R} {F1}"+'\n')

    print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
    f.write(f"all: {sum(aps)/len(aps)} {sum(ps)/len(ps)} {sum(rs)/len(rs)} {sum(f1s)/len(f1s)}" + '\n')
    f.close()



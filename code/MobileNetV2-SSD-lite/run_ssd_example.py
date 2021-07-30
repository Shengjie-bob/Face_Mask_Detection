from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import cv2
import sys
import numpy as np
import os
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if len(sys.argv) < 5:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
image_path = sys.argv[4]
output_path =sys.argv[5]

if os.path.exists(image_path):
    pass
else:
    os.mkdir(image_path)

if os.path.exists(output_path):
    pass
else:
    os.mkdir(output_path)


img_files= os.listdir(image_path)
if '.DS_Store' in img_files:
    img_files.remove('.DS_Store')


class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)


if net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of mb2-ssd-lite.")
    sys.exit(1)
net.load(model_path)
net = net.to(DEVICE)


if net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200,device=DEVICE)
else:
    print("The net type is wrong. It should be one of mb2-ssd-lite.")
    sys.exit(1)


for img_file in img_files:

    img_path = image_path+'/{}'.format(img_file)
    orig_image = cv2.imread(img_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.4)

    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(num_classes)]

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        tl = round(0.002 * (orig_image.shape[0] + orig_image.shape[1]) / 2) + 1
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

        cv2.rectangle(orig_image, c1, c2, colors[labels[i]], thickness=tl)

        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(orig_image, c1, c2, colors[labels[i]], -1, cv2.LINE_AA)
        cv2.putText(orig_image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                    lineType=cv2.LINE_AA)  # line type

    path = "{}/{}.jpg".format(output_path,img_file)
    cv2.imwrite(path, orig_image)
    print(f"Found {len(probs)} objects. The output image is {path}")

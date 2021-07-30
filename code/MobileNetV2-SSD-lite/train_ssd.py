import argparse
import os
import logging
import sys
import itertools
import numpy as np

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")


parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")

parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')

#超参数研究
parser.add_argument('--evolve',action='store_true',help='hyparameters study')
parser.add_argument('--eval_dir', default="eval_results", type=str,help='hyparameters study')

#是否对输入图像进行变换
parser.add_argument('--argument',action='store_true',help='image processing')
#是否使用adam优化器
parser.add_argument('--adam',action='store_true',help='adam optimizer')

#是否使用noise测试
parser.add_argument('--noise',action='store_true',help='add noise to train')

#保存模型的名称
parser.add_argument('--name',default='origin', type=str,help='save model name')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() and args.use_cuda else "cpu")


# Hyperparameters
hyp = {'neg_pos_ratio': 3,  # neg_pos_ratio
       'alpha': 1,  # cls loss gain
       'iou_threshold':0.5
     }

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1,alpha=1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = alpha*regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device,alpha):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    timer = Timer()

    logging.info(args)
    
    if args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    if args.argument == True:
        train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    else:
        train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std,argu=args.argument)
    if args.noise == True:
        train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std,noise=args.noise)

    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform,)
            label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        else:
            raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
        datasets.append(dataset)
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    logging.info("Prepare Validation datasets.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    elif args.dataset_type == 'open_images':
        val_dataset = OpenImagesDataset(dataset_path,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="test")
        logging.info(val_dataset)
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    logging.info("Build network.")
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    if args.adam == True:
        optimizer = torch.optim.Adam(params, lr=args.lr)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    logging.info(f"Start training from epoch {last_epoch + 1}.")

    if not args.evolve:

        for epoch in range(last_epoch + 1, args.num_epochs):
            scheduler.step()
            train(train_loader, net, criterion, optimizer,
                  device=DEVICE, debug_steps=args.debug_steps, epoch=epoch,alpha=hyp['alpha'])

            if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
                val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE,alpha=hyp['alpha'])
                logging.info(
                    f"Epoch: {epoch}, " +
                    f"Validation Loss: {val_loss:.4f}, " +
                    f"Validation Regression Loss {val_regression_loss:.4f}, " +
                    f"Validation Classification Loss: {val_classification_loss:.4f}"
                )
                model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-{args.name}.pth")
                net.save(model_path)
                logging.info(f"Saved model {model_path}")
    else:

        alphas = np.array([0.6,0.8,1,1.2,1.4])
        f = open('./{}/alpha.txt'.format(args.eval_dir),'w+')
        for i in range(len(alphas)):
            hypcopy =hyp.copy()
            hypcopy['alpha']= alphas[i]

            for epoch in range(last_epoch + 1, args.num_epochs):
                scheduler.step()
                train(train_loader, net, criterion, optimizer,
                      device=DEVICE, debug_steps=args.debug_steps, epoch=epoch, alpha=hypcopy['alpha'])

                if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
                    val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE,
                                                                                  alpha=hypcopy['alpha'])
                    logging.info(
                        f"Epoch: {epoch}, " +
                        f"Validation Loss: {val_loss:.4f}, " +
                        f"Validation Regression Loss {val_regression_loss:.4f}, " +
                        f"Validation Classification Loss: {val_classification_loss:.4f}"
                    )
                if epoch == args.num_epochs - 1:
                    f.write(f"Epoch: {epoch} alpha: {hypcopy['alpha']} valloss: {val_loss} regloss: {val_regression_loss} clcloss: {val_classification_loss}"+'\n')
                    model_path = os.path.join(args.checkpoint_folder, f"{args.net}-alpha-{hypcopy['alpha']}-Loss-{val_loss}.pth")
                    net.save(model_path)
                    logging.info(f"Saved model {model_path}")

        f.close()

        ratios = np.array([1, 3, 5, 7, 9])
        f = open('./{}/ratio.txt'.format(args.eval_dir),'w+')
        for i in range(len(ratios)):
            hypcopy = hyp.copy()
            hypcopy['neg_pos_ratio'] = ratios[i]
            #更新规则
            criterion = MultiboxLoss(config.priors, iou_threshold=hyp['iou_threshold'], neg_pos_ratio=hypcopy['neg_pos_ratio'],
                                     center_variance=0.1, size_variance=0.2, device=DEVICE)
            for epoch in range(last_epoch + 1, args.num_epochs):

                scheduler.step()
                train(train_loader, net, criterion, optimizer,
                      device=DEVICE, debug_steps=args.debug_steps, epoch=epoch, alpha=hyp['alpha'])

                if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
                    val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE,
                                                                                  alpha=hyp['alpha'])
                    logging.info(
                        f"Epoch: {epoch}, " +
                        f"Validation Loss: {val_loss:.4f}, " +
                        f"Validation Regression Loss {val_regression_loss:.4f}, " +
                        f"Validation Classification Loss: {val_classification_loss:.4f}"
                    )
                if epoch == args.num_epochs - 1:
                    f.write(f"Epoch: {epoch} ratios: {hypcopy['neg_pos_ratio']} valloss: {val_loss} regloss: {val_regression_loss} clcloss: {val_classification_loss}" + '\n')
                    model_path = os.path.join(args.checkpoint_folder, f"{args.net}-ratio-{hypcopy['neg_pos_ratio']}-Loss-{val_loss}.pth")
                    net.save(model_path)
                    logging.info(f"Saved model {model_path}")

        f.close()

        iou_t = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        f = open('./{}/iou.txt'.format(args.eval_dir),'w+')
        for i in range(len(iou_t)):
            hypcopy = hyp.copy()
            hypcopy['iou_threshold'] = iou_t[i]
            #更新规则
            criterion = MultiboxLoss(config.priors, iou_threshold=hypcopy['iou_threshold'], neg_pos_ratio=hyp['neg_pos_ratio'],
                                     center_variance=0.1, size_variance=0.2, device=DEVICE)
            for epoch in range(last_epoch + 1, args.num_epochs):

                scheduler.step()
                train(train_loader, net, criterion, optimizer,
                      device=DEVICE, debug_steps=args.debug_steps, epoch=epoch, alpha=hyp['alpha'])

                if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
                    val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE,
                                                                                  alpha=hyp['alpha'])
                    logging.info(
                        f"Epoch: {epoch}, " +
                        f"Validation Loss: {val_loss:.4f}, " +
                        f"Validation Regression Loss {val_regression_loss:.4f}, " +
                        f"Validation Classification Loss: {val_classification_loss:.4f}"
                    )
                if epoch == args.num_epochs - 1:
                    f.write(f"Epoch: {epoch} ratios: {hypcopy['iou_threshold']} valloss: {val_loss} regloss: {val_regression_loss} clcloss: {val_classification_loss}" + '\n')
                    model_path = os.path.join(args.checkpoint_folder, f"{args.net}-iou-{hypcopy['iou_threshold']}-Loss-{val_loss}.pth")
                    net.save(model_path)
                    logging.info(f"Saved model {model_path}")

        f.close()
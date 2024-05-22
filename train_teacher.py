import os
import argparse
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.imagenet import get_imagenet_dataloaders

from helper.util import adjust_learning_rate
from helper.loops import train_vanilla as train, validate


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'ResNet18', 'ResNet34', 'ResNet50',
                                 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', ])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    args = parser.parse_args()

    # set different learning rate from these 4 models
    if args.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        args.learning_rate = 0.01

    args.model_path = './save/models'
    args.tb_path = './save/tensorboard'

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = [int(it) for it in iterations]
    args.model_name = f'{args.model}_{args.dataset}_lr_{args.learning_rate}_decay_{args.weight_decay}_trial_{args.trial}'

    args.tb_folder = os.path.join(args.tb_path, args.model_name)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    return args


def main():
    best_acc = 0

    args = parse_option()

    # dataloader
    if args.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
        n_cls = 100
    elif args.dataset == 'imagenet':
        train_loader, val_loader = get_imagenet_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
        n_cls = 1000
    else:
        raise NotImplementedError(args.dataset)

    # model
    model = model_dict[args.model](num_classes=n_cls)
    model = nn.DataParallel(model)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    criterion = criterion.to(device)
    cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(1, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, args)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(args.save_folder, f'{args.model}_best.pth')
            print('saving the best model!')
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # save model
    state = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(args.save_folder, f'{args.model}_last.pth')
    torch.save(state, save_file)


if __name__ == '__main__':
    main()

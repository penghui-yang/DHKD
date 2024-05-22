"""
the general training framework
"""


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

from distiller_zoo import DistillKL, BinaryKLNorm

from helper.util import adjust_learning_rate
from helper.loops import train_distill as train, validate
from helper.optimizer import OptimGradAlign


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
    parser.add_argument('--gc', type=float, default=None, help='gradient clipping')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    parser.add_argument("--dual_head", dest="aux_classifier", action="store_true")
    parser.add_argument("--single_head", dest="aux_classifier", action="store_false")
    parser.set_defaults(aux_classifier=False)  # True: dual head, False: single head

    parser.add_argument("--linear", dest="aux_classifier_linear", action="store_true")
    parser.add_argument("--nonlinear", dest="aux_classifier_linear", action="store_false")
    parser.set_defaults(aux_classifier_linear=True)  # True: linear classifier, False: nonlinear classifier

    parser.add_argument("--ga", dest="ga", action="store_true")
    parser.add_argument("--not_ga", dest="ga", action="store_false")
    parser.set_defaults(ga=False)  # True: gradient alignment optimizer, False: common optimizer

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'dhkd'])

    parser.add_argument('-t', '--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD loss')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--BinaryKL_T', type=float, default=2, help='temperature for BinaryKL distillation')

    args = parser.parse_args()

    # set different learning rate from these models
    if args.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        args.learning_rate = args.learning_rate * 0.2

    args.model_path = './save/student_model'
    args.tb_path = './save/student_tensorboards'

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = [int(it) for it in iterations]
    args.model_t = get_teacher_name(args.path_t)

    args.model_name = f'S_{args.model_s}_T_{args.model_t}_{args.dataset}_{args.distill}_r_{args.gamma}_a_{args.alpha}_b_{args.alpha}'
    if args.distill in ['dhkd']:
        args.model_name = f'{args.model_name}_Temp_{args.BinaryKL_T}'
    if args.aux_classifier:
        if args.aux_classifier_linear:
            args.model_name = f'{args.model_name}_aux_linear'
        else:
            args.model_name = f'{args.model_name}_aux_nonlinear'
    if args.ga:
        args.model_name = f'{args.model_name}_ga'
    args.model_name = f'{args.model_name}_{args.trial}'
    print(args.model_name)

    args.tb_folder = os.path.join(args.tb_path, args.model_name)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    return args


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return f'{segments[0]}_{segments[1]}_{segments[2]}'


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def main():
    best_acc = 0

    args = parse_option()

    # tensorboard logger
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # dataloader
    if args.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
        n_cls = 100
    else:
        raise NotImplementedError(args.dataset)

    # model
    model_t = load_teacher(args.path_t, n_cls)
    model_s = model_dict[args.model_s](num_classes=n_cls, dual_head=args.aux_classifier, 
                                       aux_head_linear=args.aux_classifier_linear)
    model_t.eval()
    model_s.eval()

    module_list = nn.ModuleList([])
    trainable_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    if args.distill == 'kd':
        criterion_kd = DistillKL(args.kd_T)
    elif args.distill in ['dhkd']:
        criterion_kd = BinaryKLNorm(temperature=args.BinaryKL_T)
    else:
        raise NotImplementedError(args.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_kd)     # knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    if args.ga:
        optimizer = OptimGradAlign(optimizer)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    module_list.to(device)
    criterion_list.to(device)
    cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, args)
    print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, args)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(args.save_folder, f'{args.model_s}_best.pth')
            print('saving the best model!')
            torch.save(state, save_file)

    print('best accuracy:', best_acc)

    # save model
    state = {
        'args': args,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(args.save_folder, f'{args.model_s}_last.pth')
    torch.save(state, save_file)


if __name__ == '__main__':
    main()

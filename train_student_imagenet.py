import os
import argparse
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision

from models import model_dict

from dataset.imagenet import get_imagenet_dataloaders

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, BinaryKLNorm

from helper.loops_imagenet import train_distill as train, validate


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:6006', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')

    parser.add_argument('--save_freq', type=int, default=20, help='save ckpts each x epochs')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.set_defaults(resume=False)  # True: resume from a break point, False: train from scratch
    parser.add_argument('--ckpt_path', type=str, default=None, help='path of the checkpoint')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,60,90', help='where to decay lr, can be a list')
    parser.add_argument('--early_stop_epoch', type=int, default=50, help='early stop epoch for DHKD')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='ResNet18', choices=['ResNet18', 'MobileNetV1'])
    parser.add_argument('--model_t', type=str, default='ResNet34', choices=['ResNet34', 'ResNet50'])

    parser.add_argument("--dual_head", dest="aux_classifier", action="store_true")
    parser.add_argument("--single_head", dest="aux_classifier", action="store_false")
    parser.set_defaults(aux_classifier=True)  # True: dual head, False: single head

    parser.add_argument("--linear", dest="aux_classifier_linear", action="store_true")
    parser.add_argument("--nonlinear", dest="aux_classifier_linear", action="store_false")
    parser.set_defaults(aux_classifier_linear=True)  # True: linear classifier, False: nonlinear classifier

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'dhkd'])
    
    parser.add_argument('-t', '--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--BinaryKL_T', type=float, default=2, help='temperature for BinaryKL distillation')

    args = parser.parse_args()

    args.model_path = './save/student_model'
    args.tb_path = './save/student_tensorboards'

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = [int(it) for it in iterations]

    args.model_name = f'S_{args.model_s}_T_{args.model_t}_{args.dataset}_{args.distill}_r_{args.gamma}_a_{args.alpha}_b_{args.alpha}'
    if args.distill in ['dhkd']:
        args.model_name = f'{args.model_name}_Temp_{args.BinaryKL_T}'
    if args.aux_classifier:
        if args.aux_classifier_linear:
            args.model_name = f'{args.model_name}_aux_linear'
        else:
            args.model_name = f'{args.model_name}_aux_nonlinear'
    args.model_name = f'{args.model_name}_{args.trial}'
    print(args.model_name)

    args.tb_folder = os.path.join(args.tb_path, args.model_name)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    
    if args.resume:
        args.begin_epoch = torch.load(args.ckpt_path)["epoch"] + 1
    else:
        args.begin_epoch = 1

    return args


def main_worker(gpu, ngpus_per_node, args):
    best_acc = 0

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # tensorboard logger
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # dataloader
    if args.dataset == 'imagenet':
        train_loader, val_loader = get_imagenet_dataloaders(args)
        n_cls = 1000
    else:
        raise NotImplementedError(args.dataset)
    

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device(f"cuda:{args.gpu}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    # model
    if args.model_t == 'ResNet34':
        model_t = torchvision.models.resnet34(weights='IMAGENET1K_V1')
    elif args.model_t == 'ResNet50':
        model_t = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    model_s = model_dict[args.model_s](num_classes=n_cls, dual_head=args.aux_classifier, 
                                       aux_head_linear=args.aux_classifier_linear)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model_s.cuda(args.gpu)
                model_t.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
                model_s = torch.nn.parallel.DistributedDataParallel(model_s, device_ids=[args.gpu], find_unused_parameters=True)
                model_t = torch.nn.parallel.DistributedDataParallel(model_t, device_ids=[args.gpu], find_unused_parameters=True)
            else:
                model_s.cuda()
                model_t.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model_s = torch.nn.parallel.DistributedDataParallel(model_s, find_unused_parameters=True)
                model_t = torch.nn.parallel.DistributedDataParallel(model_t, find_unused_parameters=True)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model_s = model_s.cuda(args.gpu)
        model_t = model_t.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model_s = torch.nn.DataParallel(model_s).cuda()
        model_t = torch.nn.DataParallel(model_t).cuda()

    if args.resume:
        model_s.load_state_dict(torch.load(args.ckpt_path)["model"])

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    # criterion
    criterion_cls = nn.CrossEntropyLoss().to(device)
    if args.distill == 'kd':
        criterion_kd = DistillKL(args.kd_T)
    elif args.distill in ['dhkd']:
        criterion_kd = BinaryKLNorm(temperature=args.BinaryKL_T)
    else:
        raise NotImplementedError(args.distill)

    criterion_kd = criterion_kd.to(device)
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_kd)     # knowledge distillation loss

    # routine
    for epoch in range(args.begin_epoch, args.epochs + 1):
        if epoch >= args.early_stop_epoch:
            args.alpha = 0
        torch.cuda.empty_cache()
        adjust_learning_rate(epoch, args, optimizer)
        if args.rank == 0:
            print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, args, device)
        time2 = time.time()

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_loss', train_loss, epoch)

        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, args, device)
        
        if args.distill_drw == epoch:
            print("change")
            try:
                model_s.module.linear2.load_state_dict(model_s.module.linear.state_dict())
            except:
                model_s.module.classifier2.load_state_dict(model_s.module.classifier.state_dict())

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            logger.log_value('test_acc', test_acc, epoch)
            logger.log_value('test_loss', test_loss, epoch)
            logger.log_value('test_acc_top5', tect_acc_top5, epoch)

            # break the loop in advance when the model collapses
            if test_acc < 1.1 and epoch > 10:
                break

            state = {
                'epoch': epoch,
                'opt': args,
                'model': model_s.state_dict(),
            }
            save_file = os.path.join(args.save_folder, f'{args.model_s}_last.pth')
            torch.save(state, save_file)

            if epoch % args.save_freq == 0:
                save_file = os.path.join(args.save_folder, f'{args.model_s}_ckpt_{epoch}.pth')
                torch.save(state, save_file)

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


        if args.distill_drw == epoch:
            print("change")
            if args.model_s != "MobileNetV1":
                model_s.module.linear2.load_state_dict(model_s.module.linear.state_dict())
            else:
                model_s.module.classifier2.load_state_dict(model_s.module.classifier.state_dict())
    
    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    if args.rank == 0:
        print('best accuracy:', best_acc)

    # save model
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        state = {
            'epoch': epoch,
            'opt': args,
            'model': model_s.state_dict(),
        }
        save_file = os.path.join(args.save_folder, f'{args.model_s}_last.pth')
        torch.save(state, save_file)


def main():
    args = parse_option()

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()

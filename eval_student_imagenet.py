import os
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from models import model_dict

from dataset.imagenet import get_imagenet_dataloaders


from helper.loops_imagenet import validate


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

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='ResNet18', choices=['ResNet18', 'MobileNetV2'])
    parser.add_argument('--model_path', type=str, default=None)

    parser.add_argument("--dual_head", dest="aux_classifier", action="store_true")
    parser.add_argument("--single_head", dest="aux_classifier", action="store_false")
    parser.set_defaults(aux_classifier=True)  # True: dual head, False: single head

    parser.add_argument("--linear", dest="aux_classifier_linear", action="store_true")
    parser.add_argument("--nonlinear", dest="aux_classifier_linear", action="store_false")
    parser.set_defaults(aux_classifier_linear=True)  # True: linear classifier, False: nonlinear classifier

    args = parser.parse_args()

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
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
                model_s = torch.nn.parallel.DistributedDataParallel(model_s, device_ids=[args.gpu], find_unused_parameters=True)
            else:
                model_s.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model_s = torch.nn.parallel.DistributedDataParallel(model_s, find_unused_parameters=True)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model_s = model_s.cuda(args.gpu)
    else:
        model_s = torch.nn.DataParallel(model_s).cuda()
    
    print(type(torch.load(args.model_path)["model"]))
    model_s.load_state_dict(torch.load(args.model_path)["model"])

    # criterion
    criterion_cls = nn.CrossEntropyLoss().to(device)

    test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, args, device)


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

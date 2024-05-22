import sys
import time
import math
import torch
from torch.cuda.amp import GradScaler, autocast

from .util import AverageMeter, accuracy


def train_vanilla(epoch, train_loader, model, criterion, optimizer, args):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    scaler = GradScaler()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    end = time.time()

    for idx, data in enumerate(train_loader):

        inputs = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        train_loader_len = int(math.ceil(train_loader._size / args.world_size / args.batch_size))

        data_time.update(time.time() - end)

        inputs = inputs.float()
        inputs = inputs.to(device)
        target = target.to(device)

        # ===================forward=====================
        with autocast(dtype=torch.float32):
            output = model(inputs)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0].item(), inputs.size(0))
        top5.update(acc5[0].item(), inputs.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        # loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % args.print_freq == 0 and args.rank == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, train_loader_len, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, args, device=None):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_kd = criterion_list[1]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    scaler = GradScaler()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    end = time.time()
    for idx, data in enumerate(train_loader):

        inputs = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        train_loader_len = int(math.ceil(train_loader._size / args.world_size / args.batch_size))

        data_time.update(time.time() - end)

        inputs = inputs.float()
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # ===================forward=====================
        with autocast(dtype=torch.float32):
            logit_s = model_s(inputs)
            with torch.no_grad():
                logit_t = model_t(inputs)
            loss_cls = criterion_cls(logit_s[0], target)

        if args.distill == 'kd':
            loss_kd = criterion_kd(logit_s[0], logit_t)
        elif args.distill == 'dhkd':
            loss_kd = criterion_kd(logit_s[1], logit_t)
        else:
            raise NotImplementedError(args.distill)

        loss = args.gamma * loss_cls + args.alpha * loss_kd

        acc1, acc5 = accuracy(logit_s[0], target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0].item(), inputs.size(0))
        top5.update(acc5[0].item(), inputs.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        # loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.step()

        # ===================meters=====================
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % args.print_freq == 0 and args.rank == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, train_loader_len, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    if hasattr(args, "distributed") and args.distributed:
        top1.all_reduce()
        top5.all_reduce()
        losses.all_reduce()
    
    if args.rank == 0:
        print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args, device=None):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):

            inputs = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
            val_loader_len = int(math.ceil(val_loader._size / args.world_size / args.batch_size))

            inputs = inputs.float()
            inputs = inputs.to(device)
            target = target.to(device)

            # compute output
            with autocast(dtype=torch.float32):
                output = model(inputs)
            if isinstance(output, list):
                output = output[0]
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0].item(), inputs.size(0))
            top5.update(acc5[0].item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0 and args.rank == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, val_loader_len, batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    if hasattr(args, "distributed") and args.distributed:
        top1.all_reduce()
        top5.all_reduce()
        losses.all_reduce()
    
    if args.rank == 0:
        print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')

    return top1.avg, top5.avg, losses.avg

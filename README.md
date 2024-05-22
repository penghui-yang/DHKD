# Dual-Head Knowledge Distillation

## Requirements

The repo is tested with:

> - numpy==1.24.3
> - nvidia_dali_cuda120==1.34.0
> - tensorboard_logger==0.1.0
> - torch==2.1.2
> - torchvision==0.16.2

But it should be runnable with other PyTorch versions.

To install requirements:

```
pip install -r requirements.txt
```

It is noteworthy that we use `nvidia_dali_cuda` package to accelerate the dataloader of ImageNet. If you only want to test the performance on CIFAR-100, you can choose not to install `nvidia_dali_cuda` package and delete files whose names contain "ImageNet".

## Quick start

Quick start on CIFAR-100:

```
python train_teacher.py --model resnet32x4
python train_student.py --model_s ShuffleV2 --path_t ./save/models/resnet32x4_cifar100_lr_0.05_decay_0.0005_trial_0/resnet32x4_best.pth --dual_head --nonlinear --ga --distill dhkd -r 1 -a 1 --BinaryKL_T 2
```

Quick start on ImageNet:

Before the experiment, you need to prepare the ImageNet dataset in the folder `./data` following the example shell file given by pytorch: [file](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh).

```
CUDA_VISIBLE_DEVICES=0,1 python train_student_imagenet.py --model_t ResNet34 --model_s ResNet18 --dataset imagenet --batch_size 128 --epochs 100 --learning_rate 0.1 --lr_decay_epochs 30,60,90 --lr_decay_rate 0.1 --weight_decay 1e-4 --momentum 0.9 --distill dhkd -r 1 -a 0.1 --dual_head --linear --BinaryKL_T 2 --trial 1 --multiprocessing-distributed --world-size 1 --rank 0
```
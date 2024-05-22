# Reference to the following open-source repository:
# Repository Name: DALI
# Repository URL: https://github.com/NVIDIA/DALI
# File Paht: docs/examples/use_cases/pytorch/resnet50/main.py

import os

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn


@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels


def get_imagenet_dataloaders(args):

    data_folder = './data/imagenet'
    train_folder = os.path.join(data_folder, 'train')
    val_folder = os.path.join(data_folder, 'val')
    
    crop_size = 224
    val_size = 256

    # The Manual says:
    # If offers CPU and GPU based pipeline for DALI - use dali_cpu switch to enable CPU one. 
    # For heavy GPU networks (like RN50) CPU based one is faster, for some lighter where CPU 
    # is the bottleneck like RN18 GPU is.

    train_pipe = create_dali_pipeline(batch_size=args.batch_size,
                                      num_threads=args.num_workers,
                                      device_id=args.rank,
                                      seed=12 + args.rank,
                                      data_dir=train_folder,
                                      crop=crop_size,
                                      size=val_size,
                                      dali_cpu=False,
                                      shard_id=args.rank,
                                      num_shards=args.world_size,
                                      is_training=True)
    train_pipe.build()
    train_loader = DALIClassificationIterator(train_pipe, reader_name="Reader",
                                              last_batch_policy=LastBatchPolicy.PARTIAL,
                                              auto_reset=True)

    val_pipe = create_dali_pipeline(batch_size=args.batch_size,
                                     num_threads=args.num_workers,
                                     device_id=args.rank,
                                     seed=12 + args.rank,
                                     data_dir=val_folder,
                                     crop=crop_size,
                                     size=val_size,
                                     dali_cpu=False,
                                     shard_id=args.rank,
                                     num_shards=args.world_size,
                                     is_training=False)
    val_pipe.build()
    val_loader = DALIClassificationIterator(val_pipe, reader_name="Reader",
                                            last_batch_policy=LastBatchPolicy.PARTIAL,
                                            auto_reset=True)
    
    return train_loader, val_loader

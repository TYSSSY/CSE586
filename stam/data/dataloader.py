import torch
import argparse
from torchvision.datasets import UCF101
from functools import partial
from torchvision.transforms import transforms
from PIL import Image
import numpy as np


def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization (approximation)"
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


def vid_transform_fn(x, fn):
    return [fn(Image.fromarray(X.squeeze(dim=0).data.numpy())) for X in x]


def video_collate(batch):
    is_np = isinstance(batch[0][0][0], np.ndarray)
    T = len(batch[0][0])  # number of frames
    targets = torch.tensor([b[2] for b in batch])
    if len(batch[0]) == 3:
        extra_data = [b[1] for b in batch]
    else:
        extra_data = []
    batch_size = len(batch)
    if is_np:
        dims = (batch[0][0][0].shape[2], batch[0][0][0].shape[0], batch[0][0][0].shape[1])
        tensor_uint8_CHW = torch.empty((T * batch_size, *dims), dtype=torch.uint8)
        for i in range(batch_size):
            for t in range(T):
                tensor_uint8_CHW[i * T + t] = \
                    torch.from_numpy(batch[i][0][t]).permute(2, 0, 1)
        return tensor_uint8_CHW, targets

    else:
        dims = (batch[0][0][0].shape[0], batch[0][0][0].shape[1], batch[0][0][0].shape[2])
        tensor_float_CHW = torch.empty((T * batch_size, *dims), dtype=torch.float)
        for i in range(batch_size):
            for t in range(T):
                tensor_float_CHW[i * T + t] = batch[i][0][t]
        return tensor_float_CHW, targets


def create_dataset(args, transform, train=True):
    return UCF101(args.ucf_data_dir, args.ucf_label_dir, frames_per_clip=args.frames_per_clip,
                           step_between_clips=args.step_between_clips, train=train, transform=partial(vid_transform_fn, fn=transform))


def create_dataloader(args, train=True):
    bs = args.batch_size
    if args.input_size == 448:  # squish
        tfms = transforms.Compose(
            [transforms.Resize((args.input_size, args.input_size))])
    else:  # crop
        tfms = transforms.Compose(
            [transforms.Resize(int(args.input_size / args.val_zoom_factor)),
             transforms.CenterCrop(args.input_size)])
    tfms.transforms.append(transforms.ToTensor())
    dataset = create_dataset(args, tfms, train=train)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=True, collate_fn=video_collate, pin_memory=True, drop_last=False)
    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch STAM UCF101')
    parser.add_argument('--ucf_data_dir', type=str, default='C:/Users/Tianyi/Desktop/Courses/CSE586/Final Project/STAM-master/UCF101/UCF-101')
    parser.add_argument('--ucf_label_dir', type=str, default='C:/Users/Tianyi/Desktop/Courses/CSE586/Final Project/STAM-master/UCF101/TrainTestSplits-RecognitionTask/ucfTrainTestlist')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--val_zoom_factor', type=int, default=0.875)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--frames_per_clip', type=int, default=16)
    parser.add_argument('--frame_rate', type=float, default=1.6)
    parser.add_argument('--step_between_clips', type=int, default=1000)

    args = parser.parse_args()
    train_loader = create_dataloader(args, train=True)
    test_loader = create_dataloader(args, train=False)


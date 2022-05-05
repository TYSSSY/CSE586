import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
from stam.utils.utils import accuracy, AverageMeter
from tensorboardX import SummaryWriter
from stam.models import create_model
from stam.data.dataloader import create_dataloader
from torch.optim.lr_scheduler import StepLR
#import torch.optim as optim
#import stam.models.optimizer as optim
import os.path as osp
import time


parser = argparse.ArgumentParser(description='PyTorch STAM UCF101')
parser.add_argument('--ucf_data_dir', type=str, default='C:/Users/Tianyi/Desktop/Courses/CSE586/Final Project/STAM-master/UCF101/UCF-101')
parser.add_argument('--ucf_label_dir', type=str, default='C:/Users/Tianyi/Desktop/Courses/CSE586/Final Project/STAM-master/UCF101/TrainTestSplits-RecognitionTask/ucfTrainTestlist')
parser.add_argument('--log_dir', type=str, default='C:/Users/Tianyi/Desktop/Courses/CSE586/Final Project/STAM-master/UCF101_log/shrinked_dataset/parallel_ST_100')
parser.add_argument('--save_path', type=str, default='C:/Users/Tianyi/Desktop/Courses/CSE586/Final Project/STAM-master/UCF101_weight/shrinked_dataset/parallel_ST_100')
parser.add_argument('--model_name', type=str, default='stam_16')
parser.add_argument('--num_classes', type=int, default=11)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--val_zoom_factor', type=int, default=0.875)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--frames_per_clip', type=int, default=16)
parser.add_argument('--frame_rate', type=float, default=1.6)
parser.add_argument('--step_between_clips', type=int, default=1000)
parser.add_argument('--lr', type=int, default=3e-5)
parser.add_argument('--gamma', type=int, default=0.7)
parser.add_argument('--num_epochs', type=int, default=100)


def run_epoch(data_loader, model, is_train, optimizer, desc, cur_epoch):
    running_loss = 0.
    correct = 0
    total_samples = 0
    total_batch = len(data_loader)
    criterion = nn.CrossEntropyLoss()
    start = time.time()
    for batch_idx, (input, target) in tqdm(enumerate(data_loader), total=total_batch, desc=desc):
        input = input.cuda()
        target = target.cuda()
        with torch.set_grad_enabled(is_train) and torch.autograd.set_detect_anomaly(True):
            if is_train:
                output = model(input)
                loss = criterion(output, target)
                #lr = optim.get_epoch_lr(cur_epoch + float(batch_idx) / len(data_loader))
                #optim.set_lr(optimizer, lr)
                l2_lambda = 0.001
                l2_norm = sum(p.pow(2.0).sum()
                              for p in model.parameters())

                loss = loss + l2_lambda * l2_norm
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    output = model(input)
                    loss = criterion(output, target)

        running_loss += loss.item()
        pred = torch.max(output, 1)[1]
        total_samples += target.size(0)
        corr = (pred == target)
        correct += corr.double().sum().item()
    elapsed = time.time() - start
    accuracy = correct / total_samples * 100.
    print('\n------ loss: %.3f; accuracy: %.3f; average time: %.4f' %
          (running_loss / total_batch, accuracy, elapsed / len(data_loader)))

    return running_loss / total_batch, accuracy

def main():
    args = parser.parse_args()
    writer = SummaryWriter(args.log_dir)
    train_loader = create_dataloader(args, train=True)
    test_loader = create_dataloader(args, train=False)
    model = create_model(args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, betas=(0.9, 0.999), weight_decay=1e-3)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.7)
    best_acc = 0
    for epoch in range(args.num_epochs):
        model.train(True)
        train_loss, train_accuracy = run_epoch(train_loader, model, True, optimizer, desc="Train Epoch {}".format(epoch + 1), cur_epoch=epoch)
        print('Epoch: {} Evaluating...'.format(epoch + 1))
        writer.add_scalar('train/train_loss', train_loss, epoch + 1)
        writer.add_scalar('train/train_overall_acc', train_accuracy, epoch + 1)

        model.eval()
        test_loss, test_accuracy = run_epoch(test_loader, model, False, optimizer, desc="Final test: ", cur_epoch=epoch)
        writer.add_scalar('test/test_loss', test_loss, epoch + 1)
        writer.add_scalar('test/test_overall_acc', test_accuracy, epoch + 1)
        '''
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(model.state_dict(), args.save_path+str(epoch + 1)+'.pt')
        '''
        scheduler.step()
    writer.export_scalars_to_json(osp.join(args.log_dir, "all_scalars.json"))
    writer.close()

if __name__ == "__main__":
    main()

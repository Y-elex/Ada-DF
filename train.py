import argparse
import math
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import get_dataloaders
from model import create_model
from utils import set_random_seed, Logger, AverageMeter, generate_adaptive_LD, generate_average_weights, get_accuracy, save_checkpoint
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Training')
# train configs
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--num_classes', default=8, type=int)
parser.add_argument('--num_samples', default=30000, type=int)
# method configs
parser.add_argument('--threshold', default=0.7, type=float)
parser.add_argument('--sharpen', default=False, type=bool)
parser.add_argument('--T', default=1.2, type=float)
parser.add_argument('--alpha', default=None, type=float)
parser.add_argument('--beta', default=3, type=int)
parser.add_argument('--max_weight', default=1.0, type=float)
parser.add_argument('--min_weight', default=0.2, type=float)
parser.add_argument('--drop_rate', default=0.0, type=float)
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--label_smoothing', default=0.0, type=float)
parser.add_argument('--tops', default=0.7, type=float)
parser.add_argument('--margin_1', default=0.07, type=float)
# common configs
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--dataset', default='affectnet', type=str)
parser.add_argument('--data_path', default='./datasets/affectnet', type=str)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--device_id', default=0, type=int)

args = parser.parse_args()

device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

if args.seed is not None:
    set_random_seed(args.seed)

scaler = torch.amp.GradScaler()  # ✅ AMP加速器

def main():
    best_acc = 0
    best_epoch = 0

    logger = Logger(f'./results/log-{time.strftime("%b%d_%H-%M-%S")}.txt')
    writer = SummaryWriter()
    logger.info(args)

    LD = torch.zeros(args.num_classes, args.num_classes).to(device)
    for i in range(args.num_classes):
        LD[i] = torch.zeros(args.num_classes, device=device).fill_((1 - args.threshold) / (args.num_classes - 1)).scatter_(0, torch.tensor(i, device=device), args.threshold)

    if args.sharpen:
        LD = torch.pow(LD, 1 / args.T)
        LD = LD / LD.sum(dim=1, keepdim=True)

    model = create_model(args.num_classes, args.drop_rate).to(device)

    train_loader, test_loader = get_dataloaders(args.dataset, args.data_path, args.batch_size, args.num_workers, args.num_samples, pin_memory=True)

    criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=args.label_smoothing)
    criterion_kld = nn.KLDivLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    train_acc_list, test_acc_list = [], []
    train_loss_list, test_loss_list = [], []
    epoch_list = []

    logger.info('Start training.')
    for epoch in range(1, args.epochs + 1):
        logger.info(f'\nEpoch: {epoch}/{args.epochs} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

        train_loss, _, _, alpha_1, alpha_2 = train(train_loader, model, criterion, criterion_kld, optimizer, LD, epoch)
        _, train_acc, outputs_new, targets_new, weights_new = validate(train_loader, model, criterion, epoch, phase='train')
        LD = generate_adaptive_LD(outputs_new, targets_new, args.num_classes, args.threshold, args.sharpen, args.T)
        weights_avg, weights_max, weights_min = generate_average_weights(weights_new, targets_new, args.num_classes, args.max_weight, args.min_weight)

        test_loss, test_acc, _, _, _ = validate(test_loader, model, criterion, epoch, phase='test')

        is_best = test_acc > best_acc
        if is_best:
            best_acc, best_epoch = test_acc, epoch

        logger.info(f'α1={alpha_1:.2f}, α2={alpha_2:.2f} | Train Acc={train_acc:.2f} | Test Acc={test_acc:.2f} | Best={best_acc:.2f} @Epoch {best_epoch}')
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'class_distributions': LD.detach(),
        }, is_best)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        epoch_list.append(epoch)

        scheduler.step()

    # Save training log
    df = pd.DataFrame({
        'epoch': epoch_list,
        'train_acc': train_acc_list,
        'test_acc': test_acc_list,
        'train_loss': train_loss_list,
        'test_loss': test_loss_list
    })
    df.to_excel('ADA-DF_training_log_FERPlus.xlsx', index=False)

    # Plot final results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, train_acc_list, label='Train Accuracy')
    plt.plot(epoch_list, test_acc_list, label='Test Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.title('Epoch vs Accuracy'); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, train_loss_list, label='Train Loss')
    plt.plot(epoch_list, test_loss_list, label='Test Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Epoch vs Loss'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/ADA-DF_accuracy_loss_plot_FERPlus.png')
    plt.show()


def train(train_loader, model, criterion, criterion_kld, optimizer, LD, epoch):
    if args.alpha is not None:
        alpha_1, alpha_2 = args.alpha, 1 - args.alpha
    else:
        alpha_1 = math.exp(-(1 - epoch / args.beta)**2) if epoch <= args.beta else 1
        alpha_2 = 1 if epoch <= args.beta else math.exp(-(1 - args.beta / epoch)**2)

    model.train()
    losses = AverageMeter()
    losses_ce = AverageMeter()
    losses_kld = AverageMeter()
    losses_rr = AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train Epoch {epoch}")
    for i, (images, labels, _) in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if args.dataset.lower() in ['AffectNet', 'FER-2013', 'FERPlus']:
            batch_size, ncrops, c, h, w = images.shape
            images = images.view(-1, c, h, w)
            labels = labels.repeat_interleave(ncrops)

        with torch.cuda.amp.autocast():
            if images.ndim == 5:
              images = images[:, 0]  # 取第一张 crop
            outputs_1, outputs_2, attention_weights = model(images)
            attention_weights = attention_weights.squeeze(1)
            attn = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-6)
            attn = attn * (args.max_weight - args.min_weight) + args.min_weight
            attn = attn.unsqueeze(1)

            tops = int(args.batch_size * args.tops)
            _, top_idx = torch.topk(attention_weights, tops)
            _, low_idx = torch.topk(attention_weights, args.batch_size - tops, largest=False)
            high_mean = attention_weights[top_idx].mean()
            low_mean = attention_weights[low_idx].mean()
            diff = low_mean - high_mean + args.margin_1
            RR_loss = diff if diff > 0 else torch.tensor(0.0, device=device)

            loss_ce = criterion(outputs_1, labels).mean()
            labels_onehot = F.one_hot(labels, args.num_classes).float()
            targets = (1 - attn) * F.softmax(outputs_1, dim=1) + attn * LD[labels]
            loss_kld = criterion_kld(F.log_softmax(outputs_2, dim=1), targets).sum() / args.batch_size

            loss = alpha_2 * loss_ce + alpha_1 * loss_kld + RR_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), images.size(0))
        losses_ce.update(loss_ce.item(), images.size(0))
        losses_kld.update(loss_kld.item(), images.size(0))
        losses_rr.update(RR_loss.item(), images.size(0))

    return losses.avg, losses_ce.avg, losses_kld.avg, alpha_1, alpha_2


def validate(data_loader, model, criterion, epoch, phase='train'):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()

    outputs_all = []
    targets_all = []
    weights_all = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"{phase.capitalize()} Epoch {epoch}")
    with torch.no_grad():
        for i, (inputs, targets, _) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # 🧠 检查是否是 TenCrop 生成的 5D 输入
            is_tencrop = inputs.ndim == 5  # shape: [B, 10, 3, 224, 224]

            if is_tencrop:
                B, ncrops, C, H, W = inputs.size()
                inputs = inputs.view(-1, C, H, W)  # [B*10, 3, 224, 224]

            # 🔁 前向传播
            _, outputs, attn_weights = model(inputs)

            # 如果是 TenCrop（验证阶段），将输出 reshape 回原大小并取平均
            if is_tencrop:
                outputs = outputs.view(B, ncrops, -1).mean(1)        # 输出: [B, num_classes]
                attn_weights = attn_weights.view(B, ncrops, -1).mean(1)  # 权重也做 mean

            # 🔢 计算 loss 和 acc
            loss = criterion(outputs, targets).mean()
            top1, _ = get_accuracy(outputs, targets, topk=(1, 5))

            losses.update(loss.item(), targets.size(0))
            accs.update(top1.item(), targets.size(0))
            outputs_all.append(outputs)
            targets_all.append(targets)
            weights_all.append(attn_weights)

    return (
        losses.avg,
        accs.avg,
        torch.cat(outputs_all, dim=0),
        torch.cat(targets_all, dim=0),
        torch.cat(weights_all, dim=0),
    )



if __name__ == '__main__':
    main()

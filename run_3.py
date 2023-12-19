import argparse
import os
import sys
import time

if (slurm_submit_dir := os.environ.get('SLURM_SUBMIT_DIR', default=None)) is not None:
    sys.path.append(os.environ['SLURM_SUBMIT_DIR'])


import torch

from unitaryPE.models.image import UnitaryCCT, SinusoidalCCT, AbsoluteCCT, UnitarySeqCCT
from unitaryPE.models.image.augmentations import CIFAR10Policy
from unitaryPE.neural.schedule import make_schedule
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from typing import Literal


def run(
        model: Literal['Unitary', 'Sinusoidal', 'Absolute', 'UnitarySeq'],
        num_epochs: int,
        num_layers: int,
        num_heads: int,
        dim: int,
        mlp_ratio: int,
        dataset: Literal['cifar10', 'mnist', 'cifar100'],
        data_dir: str,
        batch_size: int,
        seed: int = 42,
        store_path: str | None = None):
    start_time = time.time()

    match dataset:
        case 'cifar10':
            augmentations = [CIFAR10Policy(),
                             transforms.RandomCrop((32, 32), padding=4),
                             transforms.RandomHorizontalFlip()]
            transformations = [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])]
            train_set = CIFAR10(data_dir, train=True, download=True,
                                transform=transforms.Compose([*augmentations, *transformations]))
            test_set = CIFAR10(data_dir, train=False, download=True,
                               transform=transforms.Compose(transformations))
            in_channels, num_classes, image_size = 3, 10, 100
        case 'cifar100':
            raise NotImplementedError
            # transform = transforms.Compose([
            #     transforms.Resize(32),
            #     transforms.CenterCrop(32),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            # train_set = CIFAR100(data_dir, train=True, transform=transform)
            # test_set = CIFAR100(data_dir, train=False, transform=transform)
            # in_channels, num_classes = 1, 100
        case 'mnist':
            raise NotImplementedError
            # train_set = MNIST(data_dir, train=True, transform=transforms.ToTensor())
            # test_set = MNIST(data_dir, train=False, transform=transforms.ToTensor())
            # in_channels, num_classes = 1, 10
        case _:
            raise ValueError

    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    torch.manual_seed(seed)

    match model:
        case 'Unitary':
            model = UnitaryCCT(
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                kernel_size=(3, 3),
                in_channels=in_channels,
                num_classes=num_classes,
                mlp_ratio=mlp_ratio).cuda()
        case 'Sinusoidal':
            model = SinusoidalCCT(
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                kernel_size=(3, 3),
                in_channels=in_channels,
                num_classes=num_classes,
                mlp_ratio=mlp_ratio).cuda()
        case 'Absolute':
            model = AbsoluteCCT(
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                kernel_size=(3, 3),
                in_channels=in_channels,
                num_classes=num_classes,
                mlp_ratio=mlp_ratio,
                num_embeddings=image_size).cuda()
        case 'UnitarySeq':
            model = UnitarySeqCCT(
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                kernel_size=(3, 3),
                in_channels=in_channels,
                num_classes=num_classes,
                mlp_ratio=mlp_ratio).cuda()
        case _:
            raise ValueError

    steps_per_epoch = len(train_dl)
    optim = AdamW(model.parameters(), lr=1, weight_decay=3e-2)
    scheduler = LambdaLR(
        optimizer=optim,
        lr_lambda=make_schedule(
            warmup_steps=steps_per_epoch * 5,
            warmdown_steps=steps_per_epoch * (num_epochs - 5),
            total_steps=steps_per_epoch * num_epochs,
            min_lr=1e-9,
            max_lr=5e-4,
            init_lr=1e-7))

    best_epoch, best_dev_acc = None, -1e10
    for epoch in range(num_epochs):
        model.train()
        print(f'{epoch}')
        epoch_loss, batch_correct, total_correct, rma = 0, 0, 0, 0
        for batch_input, target in train_dl:
            batch_input = batch_input.cuda()
            target = target.cuda()
            preds = model.forward(batch_input)
            loss = torch.nn.functional.cross_entropy(preds, target, label_smoothing=0.1)
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
            epoch_loss += loss.item()
            batch_correct = (preds.argmax(dim=-1) == target).sum()
            total_correct += batch_correct
        print(f'Train loss {epoch_loss}')
        print(f'Accuracy (m) {total_correct/len(train_set)}')
        model.eval()
        epoch_loss, total_correct = 0, 0
        with torch.no_grad():
            for batch_input, target in test_dl:
                batch_input = batch_input.cuda()
                target = target.cuda()
                preds = model.forward(batch_input)
                loss = torch.nn.functional.cross_entropy(preds, target)
                epoch_loss += loss.item()
                total_correct += (preds.argmax(dim=-1) == target).sum()
            print(f'Test loss {epoch_loss}')
            print(f'Dev acc (token) {(dev_acc := total_correct / len(test_set))}')
            if dev_acc > best_dev_acc and store_path is not None:
                best_epoch, best_dev_acc = epoch, dev_acc
                torch.save(model.state_dict(), store_path)
        sys.stdout.flush()
    duration = time.time() - start_time
    print(f'Training took {duration} seconds. Best epoch was {best_epoch}')
    sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser(description='Run a single training iteration')
    parser.add_argument('--model', type=str, choices=['Unitary', 'Sinusoidal', 'Absolute', 'UnitarySeq'], help='Type of model to use')
    parser.add_argument('--dataset', type=str, choices=['cifar10'], help='Which model to train on')
    parser.add_argument('--data_dir', type=str, help='Where is the data located')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--num_layers', type=int, default=7, help='Number of layers for the model')
    parser.add_argument('--dim', type=int, default=256, help='Dimension of the model')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--store_path', type=str, default=None, help='If/where to store the trained model')
    parser.add_argument('--seed', type=int, default=42, help='The id of the current repetition')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train with')
    parser.add_argument('--mlp_ratio', type=int, default=2, help='How big the intermediate FF dimension is')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(model=args.model,
        num_heads=args.num_heads,
        num_epochs=args.num_epochs,
        num_layers=args.num_layers,
        dim=args.dim,
        seed=args.seed,
        dataset=args.dataset,
        store_path=args.store_path,
        batch_size=args.batch_size,
        mlp_ratio=args.mlp_ratio,
        data_dir=args.data_dir)


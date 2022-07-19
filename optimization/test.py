import torch
from meta_optimizer import *
from model import *
from torchvision import datasets, transforms
import argparse
import torch.nn.functional as F
import pickle
import random
import os

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--optimizer_steps', type=int, default=100, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')
parser.add_argument('--updates_per_epoch', type=int, default=10, metavar='N',
                    help='updates per epoch (default: 100)')
parser.add_argument('--max_epoch', type=int, default=200, metavar='N',
                    help='number of epoch (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=20, metavar='N',
                    help='hidden size of the meta optimizer (default: 20)')
parser.add_argument('--num_layers', type=int, default=1, metavar='N',
                    help='number of LSTM layers (default: 1)')
parser.add_argument('--save', type=str, default='ckpt',
                   help="location to optimizer checkpoints")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)



def test_error(model, test_loader):
    test_loss = 0.0
    with torch.no_grad():
        for _, (image, label) in enumerate(test_loader):
            image, label = image.cuda(), label.cuda()
            f_x = model(image)
            loss = F.nll_loss(f_x, label)
            test_loss += loss.item()
    test_loss = test_loss / len(test_loader)

    return test_loss


def test():
    save_root = args.save
    meta_model = CifarCNN(num_class=10)
    if args.cuda:
        meta_model.cuda()
    meta_optimizer = MetaOptimizerRNN(MetaModel(meta_model), args.num_layers, 1, args.hidden_size)
    if args.cuda:
        meta_optimizer.cuda()
    optimizer_path = os.path.join(save_root, 'smoothed-optimizer.pt')
    meta_optimizer.load_state_dict(torch.load(optimizer_path))
    model = CifarCNN(num_class=10)
    if args.cuda:
        model.cuda()
    train_iter = iter(train_loader)
    loss_sum = 0.0
    step = 0
    losses = []
    tst_losses = []

    for param in meta_optimizer.parameters():
        param.requires_grad = False

    for k in range(args.optimizer_steps // args.truncated_bptt_step):
        # Keep states for truncated BPTT
        meta_optimizer.reset_lstm(keep_states=k > 0, model=model, use_cuda=args.cuda)
        # hidden = meta_optimizer.module.h0
        for j in range(args.truncated_bptt_step):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            if args.cuda:
                x, y = x.cuda(), y.cuda()

            f_x = model(x)
            loss = F.nll_loss(f_x, y)
            model.zero_grad()
            loss.backward()


            meta_model, _ = meta_optimizer.meta_update(model, loss.data)
            meta_optimizer.zero_grad()

            f_x = meta_model(x)
            loss = F.nll_loss(f_x, y)

            step += 1
            loss_sum += loss.item()
            avg_loss = loss_sum / step
            losses.append(loss.item())
            print("Step {}".format(step))
            print("Training loss: {}".format(loss.item()))
            test_loss = test_error(meta_model, test_loader)
            print("Testing loss: {}".format(test_loss))
            print("-----------------------------------\n")
            tst_losses.append(test_loss)
if __name__ == '__main__':
    test()
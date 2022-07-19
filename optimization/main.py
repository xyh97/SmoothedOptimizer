import argparse
import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from meta_optimizer import *
from model import *
from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='RNN Optimizer in PyTorch')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--optimizer_steps', type=int, default=200, metavar='N',
                    help='number of meta optimizer steps (default: 200)')
parser.add_argument('--lamb', type=float, default=1.0, metavar='N',
                    help='regularization coefficient')
parser.add_argument('--eps', type=float, default=1.0, metavar='N',
                    help='perturbation strength')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')
parser.add_argument('--updates_per_epoch', type=int, default=10, metavar='N',
                    help='updates per epoch (default: 10)')
parser.add_argument('--max_epoch', type=int, default=200, metavar='N',
                    help='number of epoch (default: 200)')
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

assert args.optimizer_steps % args.truncated_bptt_step == 0


random.seed(args.seed)
torch.manual_seed(args.seed)

# load CIFAR-10
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


def test(model, test_loader):
    test_loss = 0.0
    with torch.no_grad():
        for _, (image, label) in enumerate(test_loader):
            image, label = image.cuda(), label.cuda()
            f_x = model(image)
            loss = F.nll_loss(f_x, label)
            test_loss += loss.item()
    test_loss = test_loss / len(test_loader)

    return test_loss


def main():
    lamb = args.lamb
    eps = args.eps
    save_root = args.save
    print("Lambda {}, Epsilon {}".format(lamb, eps))
    # Create a meta optimizer that wraps a model into a meta model
    # to keep track of the meta updates.
    meta_model = CifarCNN(num_class=10)
    if args.cuda:
        meta_model.cuda()

    meta_optimizer = MetaOptimizerRNN(MetaModel(meta_model), args.num_layers, 1, args.hidden_size)
    if args.cuda:
        meta_optimizer.cuda()

    optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-4)
    best_loss = 9999.99

    for epoch in range(args.max_epoch):
        test_loss = 0.0

        train_iter = iter(train_loader)
        for i in range(args.updates_per_epoch):

            # Sample a new model
            model = CifarCNN(num_class=10)
            if args.cuda:
                model.cuda()

            for k in range(args.optimizer_steps // args.truncated_bptt_step):

                meta_optimizer.reset_lstm(
                    keep_states=k > 0, model=model, use_cuda=args.cuda)

                loss_sum = 0
                prev_loss = torch.zeros(1)
                if args.cuda:
                    prev_loss = prev_loss.cuda()
                for j in range(args.truncated_bptt_step):
                    try:
                        x, y = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        x, y = next(train_iter)
                    if args.cuda:
                        x, y = x.cuda(), y.cuda()

                    # First we need to compute the gradients of the model
                    f_x = model(x)
                    loss = F.nll_loss(f_x, y)

                    model.zero_grad()
                    loss.backward()

                    meta_model, dis = meta_optimizer.meta_update(model, loss.data, eps)

                    # Compute a loss for a step the meta optimizer
                    f_x = meta_model(x)
                    if eps is None:
                        loss = F.nll_loss(f_x, y)
                    else:
                        loss = F.nll_loss(f_x, y) + lamb * dis

                    loss_sum += (k * args.truncated_bptt_step) * (loss - Variable(prev_loss))
                    # loss_sum += (loss - Variable(prev_loss))

                    prev_loss = loss.data


                # Update the parameters of the meta optimizer
                meta_optimizer.zero_grad()
                loss_sum.backward()
                for param in meta_optimizer.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            test_loss += test(model, test_loader)

        avg_loss = test_loss / args.updates_per_epoch
        if avg_loss < best_loss:
            if not os.path.exists(save_root):
                os.mkdir(save_root)
            optimizer_path = os.path.join(save_root, 'smoothed-optimizer.pt')
            torch.save(meta_optimizer.state_dict(), optimizer_path)
            best_loss = avg_loss
        print("Epoch: {}, test loss {}".format(epoch, avg_loss))
    print("Best loss: {}".format(best_loss))

if __name__ == "__main__":
    main()

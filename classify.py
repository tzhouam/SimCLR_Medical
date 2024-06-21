import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from tqdm import tqdm
import argparse
import json
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime

parser = argparse.ArgumentParser(description='Supervised Classifier Training')
parser.add_argument('--data', metavar='DIR', default='/home/raytrack/simclr/SimCLR/dataset/finalfitz17k_label/',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='fitzpatrick',
                    help='dataset name', choices=['stl10', 'cifar10','fitzpatrick'])
parser.add_argument('--train-data-portion', default=0.1, type=float,
                    help='The portion of data used for supervised finetune. Default is 0.1.')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=['resnet18', 'resnet50', 'resnet101', 'densenet121', 'densenet169', 'densenet201'],
                    help='model architecture. (default: resnet18)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
# parser.add_argument('--seed', default=None, type=int,
#                     help='seed for initializing training. ')
# parser.add_argument('--disable-cuda', action='store_true',
#                     help='Disable CUDA')
# parser.add_argument('--fp16-precision', action='store_true',
#                     help='Whether or not to use 16-bit precision GPU training.')

# parser.add_argument('--out_dim', default=128, type=int,
#                     help='feature dimension (default: 128)')

#we added new args
parser.add_argument('--img-size', default=64, type=int, help='Image size')
parser.add_argument('--label-type', default='label', type=str,choices=['label','nine_partition_label','three_partition_label'], help='label type')
# parser.add_argument('--checkpoint', default=20, type=int, help='checkpoint frequency')
parser.add_argument('--eval-freq', default=20, type=int, help='Evaluating with test dataset after per eval_freq epochs')
# parser.add_argument('--train-mode', default='simclr_finetune', choices=['simclr_finetune', 'resnet_supervise'], help='Evaluate with pretrained simclr or train with supervised learning with ResNet')
parser.add_argument('--checkpoint-path', default='./runs/simclr/densenet121/Jun20_10-37-52/checkpoint_200.pth.tar', help='Checkpoint path of pretrained simclr. Set to None if you want to train a supervised classifier from scratch.')
# checkpoint_path = './runs/Jun03_12-03-46_u20-05/checkpoint_200.pth.tar'
# checkpoint_path = None

map_label_type_to_num = {
    'label': 114,
    'nine_partition_label': 9,
    'three_partition_label': 3
}

train_losses = []
train_top1_acces = []
train_top5_acces = []
test_top1_acces = []
test_top5_acces = []
# class LinearClassifier(nn.Module):
#     def __init__(self, hidden_dim=512, num_classes=114):
#         super(LinearClassifier, self).__init__()
#         self.fc = nn.Linear(hidden_dim, num_classes)  # Adjust the final layer according to your dataset's output classes

#     def forward(self, x):
#         return self.fc(x)

def evaluate(model, test_loader, device, criterion, writer, epoch):
    global test_top1_acces, test_top5_acces
    model.eval()
    correct = 0
    total = 0
    top1_accuracy = 0
    top5_accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for counter, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            top1, top5 = accuracy(outputs, labels, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
        
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        test_loss /= (counter + 1)
    # #         _, predicted = torch.max(outputs.data, 1)
    # #         total += labels.size(0)
    # #         correct += (predicted == labels).sum().item()
    # #         top1, top5 = accuracy(outputs, labels, topk=(1, 5))
    # #         top1_accuracy += top1[0]
    # #         top5_accuracy += top5[0]
    # # accuracy = 100 * correct / total
    # # top1_accuracy /= (counter + 1)
    # # top5_accuracy /= (counter + 1)

    # test_top1_acces.append(top1_accuracy.item())
    # test_top5_acces.append(top5_accuracy.item())
    # print(f"Top1 Test Accuracy: {top1_accuracy.item()}, Top5 Test Accuracy: {top5_accuracy.item()}\n")

    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test_top1', top1_accuracy, epoch)
    writer.add_scalar('Accuracy/test_top5', top5_accuracy, epoch)

    print(f"Test Loss: {test_loss}, Top1 Test Accuracy: {top1_accuracy}, Top5 Test Accuracy: {top5_accuracy}\n")

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":
    args = parser.parse_args()
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model (ResNet/DenseNet)
    if args.arch == 'resnet18':
        model = getattr(models, args.arch)(pretrained=False, num_classes=map_label_type_to_num[args.label_type]).to(device)
        model = models.resnet18(pretrained=False, num_classes=map_label_type_to_num[args.label_type]).to(device)
    elif args.arch == 'resnet50':
        model = models.resnet50(pretrained=False, num_classes=map_label_type_to_num[args.label_type]).to(device)
    elif args.arch == 'resnet101':
        model = models.resnet101(pretrained=False, num_classes=map_label_type_to_num[args.label_type]).to(device)
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained=False, num_classes=map_label_type_to_num[args.label_type]).to(device)
    elif args.arch == 'densenet169':
        model = models.densenet169(pretrained=False, num_classes=map_label_type_to_num[args.label_type]).to(device)
    elif args.arch == 'densenet201':
        model = models.densenet201(pretrained=False, num_classes=map_label_type_to_num[args.label_type]).to(device)

    if args.checkpoint_path and args.checkpoint_path != 'None':
        # Load the pretrained model (SimCLR feature extractor)
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        state_dict = checkpoint['state_dict']
        if args.arch.startswith('resnet'):
            for k in list(state_dict.keys()):
                if k.startswith('backbone.'):
                    if k.startswith('backbone') and not k.startswith('backbone.fc'):
                        # remove prefix
                        state_dict[k[len("backbone."):]] = state_dict[k]
                del state_dict[k]

            model.load_state_dict(state_dict, strict=False)

            # Freeze all layers but the last fc
            for name, param in model.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False

            print(f"Shape of the last layer's weights: {model.fc.weight.shape}")
            print(f"Shape of the last layer's biases: {model.fc.bias.shape}")
            # input("Press Enter to continue...")
        elif args.arch.startswith('densenet'):
            for k in list(state_dict.keys()):
                if k.startswith('backbone.'):
                    if k.startswith('backbone') and not k.startswith('backbone.classifier'):
                        # remove prefix
                        state_dict[k[len("backbone."):]] = state_dict[k]
                del state_dict[k]

            model.load_state_dict(state_dict, strict=False)

            # Freeze all layers but the last classifier layer
            for name, param in model.named_parameters():
                if name not in ['classifier.weight', 'classifier.bias']:
                    param.requires_grad = False

            print(f"Shape of the last layer's weights: {model.classifier.weight.shape}")
            print(f"Shape of the last layer's biases: {model.classifier.bias.shape}")
            # input("Press Enter to continue...")

        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        # print("Parameters with grad:", parameters)
        assert len(parameters) == 2  # {classifier_layer}.weight, {classifier_layer}.bias
    else:
        # Training from scratch, so all parameters require gradient
        print("Train ResNet from scratch")
        for name, param in model.named_parameters():
            param.requires_grad = True
    # Define optimizer for the last fully connected layer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Dataset and DataLoader
    # Load your dataset
    # Use 10% of the data for training
    full_dataset = ContrastiveLearningDataset(args.data, args, mode='finetune', usage='train')
    full_dataset = full_dataset.get_dataset(args.dataset_name, n_views=2)
    print("Full dataset len:", len(full_dataset))
    print(f"Number of full train dataset classes: {full_dataset.n_classes}")
    # input("Press Enter to continue...")
    num_train = int(args.train_data_portion * len(full_dataset))
    indices = list(range(len(full_dataset)))
    train_indices = indices[:num_train]
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    train_labels = [full_dataset.labels[i] for i in train_indices]
    n_train_classes = len(set(train_labels))
    print("Train dataset size:", len(train_dataset))
    print(f"Number of train dataset classes: {n_train_classes}")
    # input("Press Enter to continue...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # You can use the rest of the data for testing or validation
    test_dataset = ContrastiveLearningDataset(args.data, args, mode='finetune', usage='test')
    test_dataset = test_dataset.get_dataset(args.dataset_name, n_views=2)
    print("Test dataset size:", len(test_dataset))
    print(f"Number of test dataset classes: {test_dataset.n_classes}")
    # input("Press Enter to continue...")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Create tensorboard writer
    train_data_portion_str = str(args.train_data_portion).replace('.', '')
    if args.checkpoint_path != 'None':
        train_mode = 'finetune'
    else:
        train_mode = 'supervised'
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    writer = SummaryWriter(f'runs/classify/{args.arch}/{train_mode}_{train_data_portion_str}_{args.label_type}_{current_time}')

    # Train the model
    model.train()
    for epoch in range(args.epochs):
        top1_train_accuracy = 0
        top5_train_accuracy = 0
        train_loss = 0
        for counter, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # if torch.any(labels < 0) or torch.any(labels >= train_dataset.num_classes):
            #     print(f"Invalid labels in batch: {labels}")
            optimizer.zero_grad()
            outputs = model(images)
            # print("output max", outputs.max(), ", output min:", outputs.min())
            # print("labels max", labels.max(), ", labels min:", labels.min())
            # input("Press Enter to continue...")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            accs = accuracy(outputs, labels, topk=(1, 5))
            top1_train_accuracy += accs[0]
            top5_train_accuracy += accs[1]
            train_loss += loss.item()

        top1_train_accuracy /= (counter + 1)
        top5_train_accuracy /= (counter + 1)
        # train_losses.append(loss.item())
        # train_top1_acces.append(top1_train_accuracy.item())
        # train_top5_acces.append(top5_train_accuracy.item())
        # print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()}, Top1 Train Accuracy: {top1_train_accuracy.item()}, Top5 Train Accuracy: {top5_train_accuracy.item()}")
        train_loss /= (counter + 1)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train_top1', top1_train_accuracy, epoch)
        writer.add_scalar('Accuracy/train_top5', top5_train_accuracy, epoch)

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss}, Top1 Train Accuracy: {top1_train_accuracy}, Top5 Train Accuracy: {top5_train_accuracy}")

        if epoch % args.eval_freq == 0:
            evaluate(model, test_loader, device, criterion, writer, epoch)
    
    # Evaluate the model after training end
    if args.epochs % args.eval_freq != 0:
        evaluate(model, test_loader, device, criterion, writer, args.epochs)

    writer.close()


    # # Plot train loss, train acc, test acc
    # train_data_portion_str = str(args.train_data_portion).replace('.', '')
    # if args.checkpoint_path != 'None':
    #     train_mode = 'finetune'
    # else:
    #     train_mode = 'supervised'
    # # Top 1
    # epochs = list(range(len(train_losses)))
    # plt.figure()
    # fig, ax1 = plt.subplots()

    # ax1.plot(epochs, train_losses, label='Training Loss', color='tab:red')
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Training Loss', color='tab:red')
    # ax1.tick_params(axis='y', labelcolor='tab:red')
    # ax1.legend(loc='upper left')

    # ax2 = ax1.twinx()
    # ax2.plot(epochs, train_top1_acces, label='Training Accuracy', color='tab:blue')
    # ax2.set_ylabel('Top1 Accuracy', color='tab:blue')
    # ax2.tick_params(axis='y', labelcolor='tab:blue')
    # ax2.legend(loc='upper right')

    # test_epochs = list(range(0, args.epochs+1, args.eval_freq))
    # ax2.plot(test_epochs, test_top1_acces, label='Test Accuracy', color='tab:green')
    # ax2.legend(loc='upper right')

    # plt.grid()
    # plt.savefig(f'loss_acc_top1_{train_data_portion_str}_{args.arch}_{args.label_type}_{train_mode}.png')
    # plt.show()

    # # Top 5
    # epochs = list(range(len(train_losses)))
    # plt.figure()
    # fig, ax1 = plt.subplots()

    # ax1.plot(epochs, train_losses, label='Training Loss', color='tab:red')
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Training Loss', color='tab:red')
    # ax1.tick_params(axis='y', labelcolor='tab:red')
    # ax1.legend(loc='upper left')

    # ax2 = ax1.twinx()
    # ax2.plot(epochs, train_top5_acces, label='Training Accuracy', color='tab:blue')
    # ax2.set_ylabel('Top5 Accuracy', color='tab:blue')
    # ax2.tick_params(axis='y', labelcolor='tab:blue')
    # ax2.legend(loc='upper right')

    # test_epochs = list(range(0, args.epochs+1, args.eval_freq))
    # ax2.plot(test_epochs, test_top5_acces, label='Test Accuracy', color='tab:green')
    # ax2.legend(loc='upper right')

    # plt.grid()
    # plt.savefig(f'loss_acc_top5_{train_data_portion_str}_{args.arch}_{args.label_type}_{train_mode}.png')
    # plt.show()

    # # Save train loss, train acc, test acc to json file
    # data = {
    #     'epochs': args.epochs,
    #     'train_losses': train_losses,
    #     'train_top1_acces': train_top1_acces,
    #     'train_top5_acces': train_top5_acces, 
    #     'eval_freq': args.eval_freq,
    #     'test_top1_acces': test_top1_acces,
    #     'test_top5_acces': test_top5_acces
    # }

    # with open(f'output_{args.dataset_name}_{train_data_portion_str}_{args.arch}_{args.label_type}_{train_mode}.json', 'w') as file:
    #     json.dump(data, file)
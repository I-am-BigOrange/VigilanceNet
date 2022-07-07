
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import config
from model import VigilanceNet
from dataset import MyDataset


def train_model(args, model, data_loader, criterion, optimizer):
    """
    :param args: Training process parameters
    :param model: The model
    :param data_loader: Training data
    :param criterion: Loss function
    :param optimizer: The optimizer
    """
    best_model_weights = model.state_dict()

    for epoch in range(args.num_epochs):
        # print('---------------------Training(epoch: {%d})----------------------' % (epoch + 1))
        # print('Training Epoch:%3d(%d per mini-batch)' % (epoch + 1, args.batch_size))
        model.train()
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            if args.has_cuda:
                inputs1 = Variable(data['inputs1']).cuda()
                inputs2 = Variable(data['inputs2']).cuda()
                labels = Variable(data['labels']).cuda()
            else:
                inputs1 = Variable(data['inputs1'])
                inputs2 = Variable(data['inputs2'])
                labels = Variable(data['labels'])
            optimizer.zero_grad()
            if args.has_cuda:
                output, output1, output2 = model(inputs1, inputs2).cuda()
                loss = criterion(output, labels).cuda() + \
                       criterion(output1, labels).cuda() + \
                       criterion(output2, labels).cuda()
            else:
                output, output1, output2 = model(inputs1, inputs2)
                loss = criterion(output, labels) + \
                       criterion(output1, labels) + \
                       criterion(output2, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [%d]     Loss:%8.4f'
              % (epoch + 1, float(running_loss / (i+1))))
    print('Finished training')


if __name__ == '__main__':

    # Detect cuda status
    HAS_CUDA = torch.cuda.is_available()
    config.args.has_cuda = HAS_CUDA

    # Initialize training parameters
    if config.args.has_cuda:
        net = VigilanceNet().cuda()
    else:
        net = VigilanceNet()

    optimizer = optim.Adam(net.parameters(), lr=config.args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Load training set
    train_dataset = MyDataset(path=config.args.train_data_path)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=config.args.batch_size,
                                   shuffle=True,
                                   num_workers=config.args.num_workers
                                   )

    # Train the model
    train_model(config.args, net, train_data_loader, criterion, optimizer)

    # Save model weights
    state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, config.args.save_weights_path)

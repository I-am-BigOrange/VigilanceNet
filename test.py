import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable

import config
from model import VigilanceNet
from dataset import MyDataset


def test_model(model, weights, args, data_loader):
    """
    :param model: The model
    :param weights: Model weights
    :param args: Testing process parameters
    :param data_loader: Testing data
    :return rmse: The root mean square error
    :return corr: The correlation coefficient
    """
    predicted = np.zeros((0, 1))
    ground_truth = np.zeros((0, 1))

    # Load model weight
    model.load_state_dict(weights['net'])
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            if args.has_cuda:
                inputs1 = Variable(data['inputs1']).cuda()
                inputs2 = Variable(data['inputs2']).cuda()
                labels = Variable(data['labels']).cuda()
            else:
                inputs1 = Variable(data['inputs1'])
                inputs2 = Variable(data['inputs2'])
                labels = Variable(data['labels'])
            if args.has_cuda:
                outputs, _, _ = model(inputs1, inputs2).cuda()
            else:
                outputs, _, _ = model(inputs1, inputs2)

            predicted = np.r_[predicted, outputs.cpu()]
            ground_truth = np.r_[ground_truth, labels.cpu()]

    plt.plot(predicted, label='predicted')
    plt.plot(ground_truth, label='labels')
    plt.legend(loc=2)
    plt.show()

    rmse = np.sqrt(np.mean((predicted - ground_truth) ** 2))
    corr = np.corrcoef(np.squeeze(predicted), np.squeeze(ground_truth))
    corr = corr[0, 1]
    print("rmse: ", rmse)
    print("corr: ", corr)

    return rmse, corr


if __name__ == '__main__':

    # Detect cuda status
    HAS_CUDA = torch.cuda.is_available()
    config.args.has_cuda = HAS_CUDA

    # Initialize training parameters
    if config.args.has_cuda:
        net = VigilanceNet().cuda()
    else:
        net = VigilanceNet()

    # Load testing set
    test_dataset = MyDataset(path=config.args.test_data_path)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=config.args.batch_size,
                                  shuffle=False,
                                  num_workers=config.args.num_workers
                                  )
    # Load model weights
    checkpoint = torch.load(config.args.finish_weights_path)
    # Test model
    test_model(net, checkpoint, config.args, test_data_loader)

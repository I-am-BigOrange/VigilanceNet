import os
import argparse

project_path = os.getcwd()
train_dataset_path = project_path + os.sep + "data" + os.sep + "train_data.mat"
test_dataset_path = project_path + os.sep + "data" + os.sep + "test_data.mat"
save_weights_path = project_path + os.sep + "weights" + os.sep + "weights.pkl"

parser = argparse.ArgumentParser(description="Pytorch implementation of VigilanceNet")
parser.add_argument('--has-cuda', type=bool, default=False)
parser.add_argument('--train-data-path', type=str, default=train_dataset_path)
parser.add_argument('--test-data-path', type=str, default=test_dataset_path)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--print-freq', type=int, default=4)
parser.add_argument('--save-epoch-freq', type=int, default=1)
parser.add_argument('--save-weights-path', type=str, default=save_weights_path)
parser.add_argument('--finish-weights-path', type=str, default=save_weights_path, help="Training from one checkpoint")
parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume")
parser.add_argument('--network', type=str, default="VigilanceNet")
args = parser.parse_args()


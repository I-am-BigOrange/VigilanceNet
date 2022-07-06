
from scipy.io import loadmat
from torch.utils.data import Dataset


def load_data(path):
    """
    :param path: dataset path
    :return: EEG features, EOG features, PERCLOS labels
    """
    src = loadmat(path)

    data_eeg = src['eeg']
    data_eog = src['eog']
    labels = src['labels']

    return data_eeg, data_eog, labels


class MyDataset(Dataset):
    def __init__(self, path):
        data_eeg, data_eog, labels = load_data(path)
        self.inputs1 = data_eeg
        self.inputs2 = data_eog
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs1 = self.inputs1[idx]
        inputs2 = self.inputs2[idx]
        labels = self.labels[idx]

        return {'inputs1': inputs1,
                'inputs2': inputs2,
                'labels': labels
                }

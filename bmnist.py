import torch.utils.data as data
import os, os.path
import torch

class BinarizedMNIST(data.Dataset):
    """`Binarized MNIST <http://www.dmi.usherb.ca/~larocheh/mlpython/_modules/datasets/binarized_mnist.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``training.pt``
            and  ``test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
    """
    urls = [
        'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat',
        'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat',
        'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat',
    ]
    training_file = 'bmnist_train.pt'
    test_file = 'bmnist_test.pt'

    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            self.data = torch.load(os.path.join(self.root, self.training_file))
        else:
            self.data = torch.load(os.path.join(self.root, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.data[index].reshape(1,28,28)

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.test_file))


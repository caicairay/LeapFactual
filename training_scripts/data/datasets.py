from torch.utils.data import Dataset, random_split
import os
import sys
sys.path.append('../external_pkgs/Morpho-MNIST/')
from morphomnist import io, measure
import pandas as pd
import torch
from torchvision.datasets.folder import ImageFolder

class MorphoMNIST(Dataset):
    def __init__(self, split = "train", perturbation = "plain", root = "../../Morpho_MNIST/morphoMNIST_data", transform=None):
        super().__init__()
        assert split in ["train", "test"]
        assert perturbation in ["plain", "global", "local"]
        data_path = os.path.join(root, perturbation)
        split_dic = {'train': 'train', 'test': 't10k'}
        data_flnm = "{}-images-idx3-ubyte.gz".format(split_dic[split])
        label_flnm = "{}-labels-idx1-ubyte.gz".format(split_dic[split])
        csv_flnm = "{}-morpho.csv".format(split_dic[split])
        self.numpy_image = io.load_idx(os.path.join(data_path, data_flnm)).copy()
        self.labels = torch.from_numpy(io.load_idx(os.path.join(data_path, label_flnm)).copy())
        self.morphos = pd.read_csv(os.path.join(data_path, csv_flnm), sep=',', index_col = 0)
        self.transform = transform
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, idx):
        image = self.numpy_image[idx]
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
    def get_morpho(self, idx):
        return measure.Morphometrics(*self.morphos.iloc[idx])
    def measure_morpho(self, x):
        return measure.measure_image(x[0].detach().cpu().numpy(), verbose=False)
    def measure_morpho_batch(self, x, pool = None, chunksize = 100):
        return measure.measure_batch(x[:, 0].detach().cpu().numpy(), pool = pool, chunksize = chunksize)

class Galaxy10(Dataset):
    def __init__(self, split = "train", 
                 root = "../../Galaxy10/galaxy10_data", 
                 transform = None, 
                 fraction = 1., 
                 valid = None,
                 split_seed = 1234,
                 root2 = None, 
                 transform2 = None, 
                 fraction2 = 1., 
                 valid2 = None,
                 split_seed2 = 1234,
                 ):
        super().__init__()
        assert split in ["train", "test"]
        dataset = ImageFolder(root, transform=transform, is_valid_file=valid)
        if fraction < 1.:
            g = torch.Generator()
            if split_seed:
                g.manual_seed(split_seed)
            dataset, _ = random_split(dataset, [fraction, 1. - fraction], generator=g)
        if root2 is not None:
            dataset2 = ImageFolder(root2, transform = transform2, is_valid_file = valid2)
            if fraction2 < 1.:
                g = torch.Generator()
                if split_seed2:
                    g.manual_seed(split_seed2)
                dataset2, _ = random_split(dataset2, [fraction2, 1. - fraction2], generator=g)
            dataset = torch.utils.data.ConcatDataset([dataset, dataset2])
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)
    def __len__(self):
        return self.dataset.__len__()
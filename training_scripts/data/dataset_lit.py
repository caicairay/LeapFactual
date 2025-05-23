from typing import List, Optional,Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from .datasets import *
from pathlib import Path

class DatasetLit(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        name: str,
        root: str = None,
        train_root: str = None,
        test_root: str = None,
        image_size:int = 128,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        download: bool = False,
        train_fraction = 1.,
        augment = False,
        **kwargs,
    ):
        super().__init__()

        self.name = name
        self.root = root
        self.train_root = train_root
        self.test_root = test_root
        self.image_size = image_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.download = download
        self.data_mean = 0.128
        self.data_std = 0.305
        self.augm_sigma = 0.08
        self.label_idx = kwargs['label_idx'] if 'label_idx' in kwargs else None
        self.train_fraction = train_fraction
        self.augment = augment
        self.kwargs = kwargs

    def setup(self, stage: Optional[str] = None) -> None:
        # load the data
        if self.name == 'MorphoMNIST':
            data_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize(self.image_size),
                            ])
            self.train_dataset = MorphoMNIST(
                split = 'train',
                perturbation = 'plain',
                root = self.root if self.root else "../../Morpho_MNIST/morphoMNIST_data",
                transform = data_transform
            )
            self.test_dataset = MorphoMNIST(
                split = 'test',
                perturbation = 'plain',
                root = self.root if self.root else "../../Morpho_MNIST/morphoMNIST_data",
                transform = data_transform
            )
        elif self.name == 'Galaxy10':
            def check_valid(path):
                path = Path(path)
                return not path.stem.split('_')[-1] == '0'
            data_transform = transforms.Compose([
                      transforms.ToTensor(),
                      transforms.RandomAffine((-180, 180)), #, scale=(0.8, 1.2)),
                      transforms.CenterCrop(150),
                      transforms.Resize(128),
                  ])
            ce_data_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomAffine((-180, 180)), #, scale=(0.8, 1.2)),
                                transforms.Resize(self.image_size),
                            ])
            test_data_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.CenterCrop(150),
                                transforms.Resize(self.image_size),
                            ])
            self.train_dataset = Galaxy10(
                split = 'train',
                root = self.train_root if self.train_root else '../Galaxy10/galaxy10_data/image_data_train/',
                transform = data_transform,
                fraction = 1.,
                valid = None,
                root2 = self.kwargs.pop('train_root2', None),
                transform2 = ce_data_transform,
                fraction2 = self.kwargs.pop('fraction2', 1.),
                valid2 = check_valid,
                split_seed=self.kwargs.pop('split_seed', None),
                split_seed2=self.kwargs.pop('split_seed2', None),
                **self.kwargs
            )
            self.test_dataset = Galaxy10(
                split = 'test',
                root = self.test_root if self.test_root else '../Galaxy10/galaxy10_data/image_data_test/',
                transform = test_data_transform,
                **self.kwargs
            )
        else:
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
if __name__ == '__main__':
    for name in ['MorphoMNIST', 'Galaxy10']:
        print(name)
        vaedataset = DatasetLit(name=name, download=True)
        vaedataset.setup()
        test_loader = vaedataset.test_dataloader()
        data, label = next(iter(test_loader))
        print(data.max())
        print(data.shape)
        # sys.exit()
        print(label)
        plt.imshow(data[0].permute(1, 2, 0).cpu().numpy())
        plt.savefig(f'test_{name}.png')
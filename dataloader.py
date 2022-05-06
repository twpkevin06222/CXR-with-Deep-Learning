import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2


class CXRDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, img_size, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_pth = os.path.join(self.root_dir,
                                self.df['Path'].tolist()[idx])
        # since its grey scale image we only pick one channel for it
        image = cv2.imread(img_pth)[..., 0]
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = np.array(norm_func(image), dtype=np.float32)
        label = self.df['Label'].tolist()[idx]
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample


def norm_func(x, method='std'):
    """
    @param x: Input image
    @param method: [DEFAULT]'std': standardisation, 'min_max': min max normalisation
    @return: Normalised image
    """
    if method == 'std':
        x = (x-np.mean(x))/np.std(x)
    else:
        x = (x - np.min(x))/(np.max(x)-np.min(x))
    return x


transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-20, 20)),
    # transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 0.1)),
    transforms.ToTensor(),
])


transformed_ds = CXRDataset(csv_file='cxr.csv', root_dir='/home/kevinteng/Desktop/chest_xray',
                            img_size=256,  transform=transforms)
dataloader = DataLoader(transformed_ds, batch_size=16,
                        shuffle=True, num_workers=4)


for sample_batched in dataloader:
    print(sample_batched['image'].shape)
    print(torch.min(sample_batched['image']))
    print(torch.max(sample_batched['image']))
    print(sample_batched['label'].shape)
    for i in range(8):
        plt.imshow(np.squeeze(sample_batched['image'][i]), cmap='gray')
        print(sample_batched['label'][i])
        plt.show()
    break

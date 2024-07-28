import os
from PIL import Image
from torch.utils.data import Dataset

class CancerDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, mode='train'):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)
        
        if self.mode == 'train' or self.mode == 'val':
            label = int(self.dataframe.iloc[idx, 1])
            return image, label
        else:
            return image
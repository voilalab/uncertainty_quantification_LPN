import torch
from torch.utils.data import Dataset
import numpy as np
import os
import torchvision.transforms as T
import matplotlib.pyplot as plt

TRANSFORM = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.RandomApply(
            [T.RandomAffine(degrees=180, translate=None, scale=(0.9, 1.1))], p=0.8
        ),
        # T.RandomApply([
        #     T.ElasticTransform()
        # ], p=0.2),
        T.RandomApply([T.RandomPerspective()], p=0.2),
        T.RandomCrop(28),
        T.RandomApply(
            [
                T.ColorJitter(brightness=0.5, contrast=0.5),
            ],
            p=0.5,
        ),
    ]
)


class MNISTCTDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.data_dir = os.path.join(
            root, "mnist", split
        )
        print(self.data_dir)
        self.files = sorted(os.listdir(self.data_dir))
        
        self.transform = None#TRANSFORM
        self.dataset = [{"fn": fn} for fn in self.files]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.dataset[idx]["fn"])
        label = self.dataset[idx]["fn"][-5]
        # print(label)
        image = np.load(img_name)
        # plt.figure()
        # plt.imshow(image)
        # plt.savefig('/home/namhoon/learned-proximal-networks/exps/mayoct/models_mnist/temp/'+self.dataset[idx]["fn"][:-3]+'png')
        # plt.close()
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": label}


LPNDataset = MNISTCTDataset

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


class CIFAR10Dataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.data_dir = os.path.join(
            root, "cifar-10-batches-py", split
        )
        # print(self.data_dir)
        
        self.transform = None#TRANSFORM
        if split == 'train' or split == 'valid':
            split_folder = os.path.join(self.data_dir)

            # Dynamically detect all labels (folders) in the split folder
            self.labels = [folder for folder in os.listdir(split_folder) if os.path.isdir(os.path.join(split_folder, folder))]
            self.sorted_labels = sorted(self.labels)  # Sort labels alphabetically

            print(f"Detected labels: {self.sorted_labels}")

            # Initialize dataset
            self.dataset = []

            # Loop through each label folder
            for label_idx, label_name in enumerate(self.sorted_labels):
                label_folder = os.path.join(split_folder, label_name)
                print(f"Processing folder: {label_folder}")
                if os.path.isdir(label_folder):  # Check if it's a valid directory
                    for file in os.listdir(label_folder):
                        if file.endswith(".npy"):  # Only consider .npy files
                            self.dataset.append({
                                "img_path": os.path.join(label_folder, file),
                                "label": label_idx
                            })

            print(f"Total files loaded: {len(self.dataset)}")
      
        # self.labels = ['airplane', 'automobile', 'bird', 'ship', 'truck', 'cat', 'deer', 'dog', 'frog', 'horse']
        # self.sorted_labels = sorted(self.labels)
        # if split == 'train':
        #     self.dataset = []
        #     for label_idx, label_name in enumerate(self.labels):
                
        #         label_folder = os.path.join(self.data_dir, label_name)
        #         print(label_folder)
        #         if os.path.isdir(label_folder):
        #             for file in os.listdir(label_folder):
        #                 if file.endswith(".npy"):
        #                     self.dataset.append({"img_path": os.path.join(label_folder, file), "label": label_idx})
        #     # idx= 6 
        #     # if split == 'train': self.labels = [self.labels[idx]]
        #     # else: self.labels = self.labels.pop(idx)
        #     # self.dataset = []
        #     # for label_idx, label_name in enumerate(self.labels):
        #     #     label_folder = os.path.join(self.data_dir, label_name)
        #     #     print(label_folder)
        #     #     if os.path.isdir(label_folder):
        #     #         for file in os.listdir(label_folder):
        #     #             if file.endswith(".npy"):
        #     #                 self.dataset.append({"img_path": os.path.join(label_folder, file), "label": label_idx})
        # elif split == 'valid':

        else:
            self.labels = ['airplane', 'automobile', 'bird', 'ship', 'truck', 'cat', 'deer', 'dog', 'frog', 'horse']
            self.sorted_labels = sorted(self.labels)
            self.files = sorted(os.listdir(self.data_dir))
            self.dataset = [{"fn": fn} for fn in self.files]
            self.dataset = []
            # print(self.sorted_labels)
            cnt = 0
            idx = 0
            for file in self.files:
                # print(file)
                self.dataset.append({"img_path": os.path.join(self.data_dir, file), "label": self.sorted_labels[idx]})
                # print(f'{cnt} {idx} {self.sorted_labels[idx]}')
                cnt += 1
                if cnt % 10 == 0:
                    idx += 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset[idx]["img_path"]
        label = self.dataset[idx]["label"]

        # Load the image
        image = np.load(img_path)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        # image = image.transpose(0,2)
        # print(image.shape)
        image = image.unsqueeze(0)  # Add channel dimension if needed

        # Apply transformation (if any)
        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": label}


LPNDataset = CIFAR10Dataset

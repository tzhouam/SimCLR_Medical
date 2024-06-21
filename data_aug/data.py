import os
import glob
import torch

from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GlobImageDataset(Dataset):
    def __init__(self, root, mode, phase, img_size, transform=None,n_views=2,label_type='label'):
        self.root = root
        self.img_size = img_size
        # Set random seed to ensure every time the dataset split is the same
        random.seed(1)
        self.total_imgs = sorted([os.path.join(root, img) for img in os.listdir(root) if '.jpg' in img])
        random.shuffle(self.total_imgs)
        #label is embeded in the image name separated by '-', by order they are nine_partition_label, three_partition_label, label. Here the 'label' is the label of the illness for the image
        self.mode = mode
        if phase == 'train':
            self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * 0.7)]
        elif phase == 'val':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.7):int(len(self.total_imgs) * 0.85)]
        elif phase == 'test':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.85):]
        else:
            pass
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()
        self.n_views = n_views
        self.label_type = label_type
        self.label_idx = {'nine_partition_label':-3,'three_partition_label':-2,'label':-1}

        # Calculate n_classes
        self.labels = [int(img.split('-')[self.label_idx[self.label_type]].split('.')[0]) for img in self.total_imgs]
        self.n_classes = len(set(self.labels))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        if self.mode == 'pretrain':
            # image = image.resize((self.img_size, self.img_size)) #disable resize since we have crop in transform
            img_res = []
            label_res = []
            for i in range(self.n_views):
                img = self.transform(image)
                img_res.append(img)
                label_res.append(int(img_loc.split('-')[self.label_idx[self.label_type]].split('.')[0]))  #xxxx-1-0-45.jpg ->45, if label type is "label"
            # tensor_image1 = self.transform(image)
            # tensor_image2 = self.transform(image)
            return img_res, label_res[0]
        else: # self.mode = 'finetune'
            image = image.resize((self.img_size, self.img_size))
            image = transforms.ToTensor()(image)
            label = int(img_loc.split('-')[self.label_idx[self.label_type]].split('.')[0])
            if label < 0 or label >= self.n_classes:
                print(f"Invalid label {label} found in {img_loc}")
            return image, label
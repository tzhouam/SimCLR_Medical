import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm, trange


from pathlib import Path

# Define the current and new file names
# current_file_path = Path('old_file_name.txt')
# new_file_path = Path('new_file_name.txt')

# try:
#     # Rename the file
#     current_file_path.rename(new_file_path)
#     print(f"File renamed from {current_file_path} to {new_file_path}")
# except FileNotFoundError:
#     print(f"The file {current_file_path} does not exist.")
# except PermissionError:
#     print(f"Permission denied. Cannot rename {current_file_path}.")
# except Exception as e:
#     print(f"An error occurred: {e}")

dataset_root = '/home/raytrack/simclr/SimCLR/dataset/finalfitz17k_label/'
images = set([img for img in os.listdir(dataset_root) if '.jpg' in img])
image_roots = [os.path.join(dataset_root, img) for img in images]
image_features = pd.read_csv('/home/raytrack/simclr/SimCLR/fitzpatrick17k.csv')
length = len(image_features)
label_mapping = {category: idx for idx, category in enumerate(image_features['label'].unique())}
nine_mapping = {category: idx for idx, category in enumerate(image_features['nine_partition_label'].unique())}
three_mapping = {category: idx for idx, category in enumerate(image_features['three_partition_label'].unique())}


for i in tqdm(range(length),total=length):
    new_file_name = image_features.iloc[i]['md5hash'] +'-'+str(nine_mapping[image_features.iloc[i]['nine_partition_label']])\
        +'-'+str(three_mapping[image_features.iloc[i]['three_partition_label']])\
        +'-'+str(label_mapping[image_features.iloc[i]['label']])\
        +'.jpg'
    new_file_path = os.path.join(dataset_root,new_file_name)
    old_file_name = image_features.iloc[i]['md5hash']+ '.jpg'
    old_file_path = os.path.join(dataset_root,old_file_name)
    # print(new_file_path,old_file_path)
    if old_file_name not in images:
        continue
    try:
        os.rename(old_file_path,new_file_path)
    except Exception as e:
        print(f"An error occurred: {e}")
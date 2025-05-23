import h5py
from PIL import Image
import os
from tqdm import tqdm

num_files = [1081, 1853, 2645, 2027, 334, 2043,1829 ,2628 ,1423 ,1873 ]
current_num = [0] * 10

with h5py.File('Galaxy10_DECals.h5', 'r') as file:
    imgs = file['images']
    labels = file['ans']
    pbar = tqdm(range(len(labels)))
    for i in pbar:
        img = Image.fromarray(imgs[i])
        label = labels[i]
        current_num[label] += 1
        if current_num[label] > num_files[label] // 10: # training set
            savepath = f'image_data_train/ans_{label}'
        else: # test set
            savepath = f'image_data_test/ans_{label}'
        os.makedirs(savepath, exist_ok=True)
        img.save(os.path.join(savepath, f'img_{i}.jpeg'))
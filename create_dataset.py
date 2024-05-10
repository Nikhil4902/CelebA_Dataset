import os
import tqdm
import random
from collections import defaultdict

random.seed(12345)

os.system("rm -rf Datasets")
os.system("mkdir Datasets")
os.system("mkdir Datasets/train_dataset")
os.system("mkdir Datasets/validation_dataset")
os.system("mkdir Datasets/test_dataset")

num_celebs = 200
num_imgs = 30
validation_ratio = 0.1
test_ratio = 0.1
split_idx_1 = int(num_imgs * (1 - validation_ratio - test_ratio))
split_idx_2 = int(num_imgs * (1 - test_ratio))

for i in range(num_celebs):
    os.system(f"mkdir Datasets/train_dataset/celeb_{i}")
    os.system(f"mkdir Datasets/validation_dataset/celeb_{i}")
    os.system(f"mkdir Datasets/test_dataset/celeb_{i}")

d = defaultdict(list)

with open("identity_CelebA.txt") as f:
    lines = f.readlines()
    for line in lines:
        a = line.split()
        d[int(a[-1])].append(a[0])
    k = [i for i in d if len(d[i]) == num_imgs]
    random.shuffle(k)

    for i, celeb in enumerate(tqdm.tqdm(k[:num_celebs])):
        imgs = d[celeb]
        random.shuffle(imgs)
        train_imgs, validation_imgs, test_imgs = imgs[:split_idx_1], imgs[split_idx_1: split_idx_2], imgs[split_idx_2:]
        for img in train_imgs:
            os.system(f"cp img_align_celeba/{img} Datasets/train_dataset/celeb_{i}/{img}")
        for img in validation_imgs:
            os.system(f"cp img_align_celeba/{img} Datasets/validation_dataset/celeb_{i}/{img}")
        for img in test_imgs:
            os.system(f"cp img_align_celeba/{img} Datasets/test_dataset/celeb_{i}/{img}")
import os
import cv2
import cv2.data
import tqdm
import random
from collections import defaultdict

random.seed(54321)

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

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
    else:
        face_roi = gray
    return cv2.resize(face_roi, (150, 150), interpolation=cv2.INTER_LINEAR)

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
            bw_face = detect_face(f"img_align_celeba/{img}")
            cv2.imwrite(f"Datasets/train_dataset/celeb_{i}/{img}", bw_face)
            # os.system(f"cp img_align_celeba/{img} Datasets/train_dataset/celeb_{i}/{img}")
        for img in validation_imgs:
            bw_face = detect_face(f"img_align_celeba/{img}")
            cv2.imwrite(f"Datasets/validation_dataset/celeb_{i}/{img}", bw_face)
            # os.system(f"cp img_align_celeba/{img} Datasets/validation_dataset/celeb_{i}/{img}")
        for img in test_imgs:
            bw_face = detect_face(f"img_align_celeba/{img}")
            cv2.imwrite(f"Datasets/test_dataset/celeb_{i}/{img}", bw_face)
            # os.system(f"cp img_align_celeba/{img} Datasets/test_dataset/celeb_{i}/{img}")
import os
import random
import numpy as np
import PIL.Image as Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def get_data_list(src_path):
    root = 'data'
    training_list = []
    test_list = []
    f1 = open(os.path.join(root + '/training_list.txt'), 'w+')
    f2 = open(os.path.join(root + '/test_list.txt'), 'w+')

    i = 0
    j=0
    for file_path in os.listdir(os.path.join(root + '/right')):
        if i == 3000:
            break
        if i % 10 == 0:
            f2.write(root + '/right/' + str(file_path) + '\t' + 'R' + '\n')
            i += 1
            continue
        f1.write(root + '/right/' + str(file_path) + '\t' + 'R' + '\n')
        i += 1

    i = 0
    for file_path in os.listdir(os.path.join(root + '/wrong')):
        if i == 3000:
            break
        if i % 10 == 0:
            f2.write(root + '/wrong/' + str(file_path) + '\t' + 'W' + '\n')
            i += 1
            continue
        f1.write(root + '/wrong/' + str(file_path) + '\t' + 'W' + '\n')
        i += 1








class my_dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(my_dataset, self).__init__()
        self.data_path = root
        self.img_paths = []
        self.labels = []
        self.train = train

        if self.train:
            with open(os.path.join(self.data_path + '/training_list.txt'), "r", encoding="utf-8") as f:
                self.info = f.readlines()
            for img_info in self.info:
                img_path, label = img_info.strip().split('\t')
                self.img_paths.append(img_path)
                if str(label) == 'R':
                    self.labels.append(1)
                else:
                    self.labels.append(0)
        else:
            with open(os.path.join(self.data_path + '/test_list.txt'), "r", encoding="utf-8") as f:
                self.info = f.readlines()
            for img_info in self.info:
                img_path, label = img_info.strip().split('\t')
                self.img_paths.append(img_path)
                if str(label)=='R':
                    self.labels.append(1)
                else:
                    self.labels.append(0)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        #img = img.resize((28, 28), Image.ANTIALIAS)
        #img = img.convert('L')
        #img = np.array(img).astype('float32')/255       #归一化处理数据，可能便于学习
        #img = np.expand_dims(img, 0)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        #img = img.resize((127,127), Image.BILINEAR)
            img = img.resize((127, 127), Image.BILINEAR)
        img = np.array(img).astype('float32')
        img = img.transpose((2, 0, 1)) / 255
        label = self.labels[index]
        label = np.array([label], dtype="int64")
        return img, label

    def print_sample(self, index: int = 0):
        print("文件名", self.img_paths[index], "\t标签值", self.labels[index])

    def __len__(self):
        return len(self.img_paths)


src_path = "data"
training_list_path = "data/training_list.txt"
test_list_path = "data/test_list.txt"
training_batch_size = 100
test_batch_size = 20

with open(training_list_path, 'w') as f:
    f.seek(0)
    f.truncate()
with open(test_list_path, 'w') as f:
    f.seek(0)
    f.truncate()

get_data_list(src_path)

training_dataset = my_dataset(
    src_path, train=True, transform=transforms.ToTensor()
)
train_dataloader = DataLoader(
    dataset=training_dataset,
    batch_size=training_batch_size,
    shuffle=True,
)

test_dataset = my_dataset(
    src_path, train=False, transform=transforms.ToTensor()
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=test_batch_size,
    shuffle=True
)

#print(training_dataset.__getitem__(0)[0].shape)
#检查是否将数据归一化
#training_dataset.print_sample(5)
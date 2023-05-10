import torch
import numpy as np
import os
from PIL import Image, TarIO
import pickle
import tarfile

class cub200(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        super(cub200, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform


        if self._check_processed():
            print('Train file has been extracted' if self.train else 'Test file has been extracted')
        else:
            self._extract()

        if self.train:
            self.train_data, self.train_label = pickle.load(
                open(os.path.join(self.root, 'processed/train.pkl'), 'rb')
            )
        else:
            self.test_data, self.test_label = pickle.load(
                open(os.path.join(self.root, 'processed/test.pkl'), 'rb')
            )

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            img, label = self.train_data[idx], self.train_label[idx]
        else:
            img, label = self.test_data[idx], self.test_label[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def _check_processed(self):
        assert os.path.isdir(self.root) == True
        assert os.path.isfile(os.path.join(self.root, 'CUB_200_2011.tgz')) == True
        return (os.path.isfile(os.path.join(self.root, 'processed/train.pkl')) and
                os.path.isfile(os.path.join(self.root, 'processed/test.pkl')))

    def _extract(self):
        processed_data_path = os.path.join(self.root, 'processed')
        if not os.path.isdir(processed_data_path):
            os.mkdir(processed_data_path)

        cub_tgz_path = os.path.join(self.root, 'CUB_200_2011.tgz')
        images_txt_path = 'CUB_200_2011/images.txt'
        train_test_split_txt_path = 'CUB_200_2011/train_test_split.txt'

        tar = tarfile.open(cub_tgz_path, 'r:gz')
        images_txt = tar.extractfile(tar.getmember(images_txt_path))
        train_test_split_txt = tar.extractfile(tar.getmember(train_test_split_txt_path))
        if not (images_txt and train_test_split_txt):
            print('Extract image.txt and train_test_split.txt Error!')
            raise RuntimeError('cub-200-1011')

        images_txt = images_txt.read().decode('utf-8').splitlines()
        train_test_split_txt = train_test_split_txt.read().decode('utf-8').splitlines()

        id2name = np.genfromtxt(images_txt, dtype=str)
        id2train = np.genfromtxt(train_test_split_txt, dtype=int)
        print('Finish loading images.txt and train_test_split.txt')
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        print('Start extract images..')
        cnt = 0
        train_cnt = 0
        test_cnt = 0
        for _id in range(id2name.shape[0]):
            cnt += 1

            image_path = 'CUB_200_2011/images/' + id2name[_id, 1]
            image = tar.extractfile(tar.getmember(image_path))
            if not image:
                print('get image: '+image_path + ' error')
                raise RuntimeError
            image = Image.open(image)
            label = int(id2name[_id, 1][:3]) - 1

            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image)
            image.close()

            if id2train[_id, 1] == 1:
                train_cnt += 1
                train_data.append(image_np)
                train_labels.append(label)
            else:
                test_cnt += 1
                test_data.append(image_np)
                test_labels.append(label)
            if cnt%1000 == 0:
                print('{} images have been extracted'.format(cnt))
        print('Total images: {}, training images: {}. testing images: {}'.format(cnt, train_cnt, test_cnt))
        tar.close()
        pickle.dump((train_data, train_labels),
                    open(os.path.join(self.root, 'processed/train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels),
                    open(os.path.join(self.root, 'processed/test.pkl'), 'wb'))


class cub200_10(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        super(cub200_10, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform


        if self._check_processed():
            print('Train file has been extracted' if self.train else 'Test file has been extracted')
        else:
            self._extract()

        if self.train:
            self.train_data, self.train_label = pickle.load(
                open(os.path.join(self.root, 'processed_10/train.pkl'), 'rb')
            )
        else:
            self.test_data, self.test_label = pickle.load(
                open(os.path.join(self.root, 'processed_10/test.pkl'), 'rb')
            )

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            img, label = self.train_data[idx], self.train_label[idx]
        else:
            img, label = self.test_data[idx], self.test_label[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def _check_processed(self):
        assert os.path.isdir(self.root) == True
        assert os.path.isfile(os.path.join(self.root, 'CUB_200_2011.tgz')) == True
        return (os.path.isfile(os.path.join(self.root, 'processed_10/train.pkl')) and
                os.path.isfile(os.path.join(self.root, 'processed_10/test.pkl')))

    def _extract(self):

        counters = [0] * 200

        processed_data_path = os.path.join(self.root, 'processed_10')
        if not os.path.isdir(processed_data_path):
            os.mkdir(processed_data_path)

        cub_tgz_path = os.path.join(self.root, 'CUB_200_2011.tgz')
        images_txt_path = 'CUB_200_2011/images.txt'
        train_test_split_txt_path = 'CUB_200_2011/train_test_split.txt'

        tar = tarfile.open(cub_tgz_path, 'r:gz')
        images_txt = tar.extractfile(tar.getmember(images_txt_path))
        train_test_split_txt = tar.extractfile(tar.getmember(train_test_split_txt_path))
        if not (images_txt and train_test_split_txt):
            print('Extract image.txt and train_test_split.txt Error!')
            raise RuntimeError('cub-200-1011')

        images_txt = images_txt.read().decode('utf-8').splitlines()
        train_test_split_txt = train_test_split_txt.read().decode('utf-8').splitlines()

        id2name = np.genfromtxt(images_txt, dtype=str)
        id2train = np.genfromtxt(train_test_split_txt, dtype=int)
        print('Finish loading images.txt and train_test_split.txt')
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        print('Start extract images..')
        cnt = 0
        train_cnt = 0
        test_cnt = 0
        for _id in range(id2name.shape[0]):
            cnt += 1

            image_path = 'CUB_200_2011/images/' + id2name[_id, 1]
            image = tar.extractfile(tar.getmember(image_path))
            if not image:
                print('get image: '+image_path + ' error')
                raise RuntimeError
            image = Image.open(image)
            label = int(id2name[_id, 1][:3]) - 1

            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image)
            image.close()

            if id2train[_id, 1] == 1:
                if counters[int(label)] < 10:
                    train_cnt += 1
                    train_data.append(image_np)
                    train_labels.append(label)
                    counters[int(label)] = counters[int(label)] + 1
            else:
                test_cnt += 1
                test_data.append(image_np)
                test_labels.append(label)
            if cnt%1000 == 0:
                print('{} images have been extracted'.format(cnt))
        print('Total images: {}, training images: {}. testing images: {}'.format(cnt, train_cnt, test_cnt))
        tar.close()
        pickle.dump((train_data, train_labels),
                    open(os.path.join(self.root, 'processed_10/train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels),
                    open(os.path.join(self.root, 'processed_10/test.pkl'), 'wb'))

import os

import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_file_from_google_drive


class Cub2011(VisionDataset):
    """`CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Cub2011, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_file_from_google_drive(self.file_id, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


if __name__ == '__main__':
    train_dataset = Cub2011('./cub2011', train=True, download=False)
    test_dataset = Cub2011('./cub2011', train=False, download=False)
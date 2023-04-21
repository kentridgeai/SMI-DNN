import keras
import numpy as np
import random
import imageio
from collections import namedtuple

def preprocessing(cfg, X_train, X_test, y_train, y_test):
    if cfg['model'] == 'MLP':
        X_train = np.reshape(X_train, [X_train.shape[0], -1]) / 255.
        X_test = np.reshape(X_test, [X_test.shape[0], -1]) / 255.
    elif cfg['model'] in ['CNN_Global']:
        if cfg['dataset'] in ['MNIST', 'Fashion_MNIST']:
            X_train = np.expand_dims(X_train, 3)
            X_test  = np.expand_dims(X_test, 3)
        X_train = X_train / 255.
        X_test = X_test / 255.
    elif cfg['model'] == 'VGG16':
        X_train = keras.applications.vgg16.preprocess_input(X_train)
        X_test = keras.applications.vgg16.preprocess_input(X_test)
        X_train = X_train / 255.
        X_test = X_test / 255.
    elif cfg['model'] in ['ResNet50']:
        X_train = keras.applications.resnet.preprocess_input(X_train)
        X_test = keras.applications.resnet.preprocess_input(X_test)
        X_train = X_train / 255.
        X_test = X_test / 255.
    if cfg['noise_ratio'] > 0.0:
        noisy_label = []
        idx = list(range(X_train.shape[0]))
        random.shuffle(idx)
        num_noise = int(cfg['noise_ratio']*X_train.shape[0])
        noise_idx = idx[:num_noise]
        for i in range(X_train.shape[0]):
            if i in noise_idx:
                noisylabel = random.randint(0, np.max(y_train))
                noisy_label.append(noisylabel)
            else:
                noisy_label.append(y_train[i])
        Y_train = keras.utils.to_categorical(noisy_label)
        Y_test = keras.utils.to_categorical(y_test)
    else:
        Y_train = keras.utils.to_categorical(y_train)
        Y_test = keras.utils.to_categorical(y_test)
    return X_train, X_test, Y_train, Y_test

def get_id_dictionary(path):
    id_dict = {}
    for i, line in enumerate(open( path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict

def get_data(id_dict, path):
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    for key, value in id_dict.items():
        train_data += [imageio.imread( path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), pilmode='RGB') for i in range(500)]
        train_labels += [value]*500
    for line in open( path + 'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        test_data.append(imageio.imread( path + 'val/images/{}'.format(img_name), pilmode='RGB'))
        test_labels.append(id_dict[class_id])
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

def shuffle_data(train_data, train_labels ):
    np.random.seed(1234)
    size = len(train_data)
    train_idx = np.arange(size)
    np.random.shuffle(train_idx)
    return train_data[train_idx], train_labels[train_idx]

def get_dataset(cfg):
    if cfg['dataset'] == 'MNIST':
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        X_train, X_test, Y_train, Y_test = preprocessing(cfg, X_train, X_test, y_train, y_test)
    elif cfg['dataset'] == 'Fashion_MNIST':
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
        X_train, X_test, Y_train, Y_test = preprocessing(cfg, X_train, X_test, y_train, y_test)
    elif cfg['dataset'] == 'CIFAR10':
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
        X_train, X_test, Y_train, Y_test = preprocessing(cfg, X_train, X_test, y_train, y_test)
    elif cfg['dataset'] == 'CIFAR100':
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
        X_train, X_test, Y_train, Y_test = preprocessing(cfg, X_train, X_test, y_train, y_test)
    Dataset = namedtuple('Dataset',['X','Y','y'])
    trn = Dataset(X_train, Y_train, y_train)
    tst = Dataset(X_test , Y_test, y_test)
    return trn, tst

        

        

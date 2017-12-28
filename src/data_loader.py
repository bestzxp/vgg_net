import os
import numpy as np
import cv2

classes = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed',
           'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize',
           'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

class DataLoader(object):
    def __init__(self):
        self.classes = classes
        self.images_urls, self.labels = self.load_all_data()
        self.cursor = 0


    def load_all_data(self):
        img_path_urls = []
        label = []
        for i in range(len(classes)):
            images = os.listdir('../data/train/{}/'.format(classes[i]))
            # print(images)
            img_path_urls.extend(['../data/train/{}/'.format(classes[i]) + image for image in images])
            label.extend([i]*len(images))
            print('class {} has {} images'.format(classes[i], len(images)))
        print('\ntotal images num: {}'.format(len(img_path_urls)))
        img_path_urls = np.array(img_path_urls)
        label = np.array(label)
        perm = np.arange(len(img_path_urls))
        np.random.shuffle(perm)
        print(img_path_urls[0])
        return img_path_urls[perm], label[perm]

    def get_batch_data(self, batch_size):
        images = np.zeros([batch_size, 320, 320, 3])
        labels = np.zeros([batch_size, len(self.classes)])
        for i in range(batch_size):
            if self.cursor >= self.images_urls.shape[0]:
                perm = np.arange(len(self.images_urls))
                np.random.shuffle(perm)
                self.images_urls = self.images_urls[perm]
                self.labels = self.labels[perm]
                self.cursor = 0
            images[i, :] = self.get_image(self.images_urls[i])
            labels[i, :] = self.one_hot(self.labels[i], len(classes))
        return images, labels

    def get_image(self, image_url):
        img = cv2.imread(image_url)
        img = cv2.resize(img, (320, 320))
        return img
    def one_hot(self, i, num_classes):
        label = np.zeros((1, num_classes))
        label[0, i] = 1
        return label


import os
import numpy as np
import cv2

classes = range(1, 129)

class DataLoader(object):
    def __init__(self):
        self.classes = classes
        self.images_urls, self.labels = self.load_all_data()
        self.valid_urls, self.valid_labels = self.load_all_data(type='valid')
        self.cursor = 0
        self.avg = np.array([[[52.7398061476, 73.6611408516, 83.7255207858]]])

        self.valid_cursor = 0


    def load_all_data(self, type='train'):
        img_path_urls = []
        label = []
        for i in range(len(classes)):
            images = os.listdir('../data/{}/{}/'.format(type, classes[i]))
            # print(images)
            img_path_urls.extend(['../data/{}/{}/'.format(type, classes[i]) + image for image in images])
            label.extend([i]*len(images))
            # print('class {} has {} images'.format(classes[i], len(images)))
        print('\ntotal images num: {}'.format(len(img_path_urls)))
        img_path_urls = np.array(img_path_urls)
        label = np.array(label)
        perm = np.arange(len(img_path_urls))
        np.random.shuffle(perm)
        return img_path_urls[perm], label[perm]

    def get_batch_data(self, batch_size):
        images = np.zeros([batch_size, 224, 224, 3])
        labels = np.zeros([batch_size, len(self.classes)])
        for i in range(batch_size):
            if self.cursor < self.images_urls.shape[0]:
                images[i, :] = (self.get_image(self.images_urls[self.cursor]) - self.avg)
                labels[i, :] = self.one_hot(self.labels[self.cursor], len(classes))
                self.cursor += 1
        return images, labels

    def get_valid_batch_data(self, batch_size):
        images = np.zeros([batch_size, 224, 224, 3])
        labels = np.zeros([batch_size, len(self.classes)])
        for i in range(batch_size):
            if self.valid_cursor < self.valid_urls.shape[0]:
                images[i, :] = (self.get_image(self.valid_urls[self.valid_cursor]) - self.avg)
                labels[i, :] = self.one_hot(self.valid_labels[self.valid_cursor], len(classes))
                self.valid_cursor += 1
        return images, labels


    def shuffle(self):
        perm = np.arange(len(self.images_urls))
        np.random.shuffle(perm)
        self.images_urls = self.images_urls[perm]
        self.labels = self.labels[perm]
        self.cursor = 0

    def get_image(self, image_url):
        # print(image_url)
        img = cv2.imread(image_url)
        img = cv2.resize(img, (224, 224))
        return img

    def one_hot(self, i, num_classes):
        label = np.zeros((1, num_classes))
        label[0, i] = 1
        return label

    # def preprocess(self, image):

if __name__ == '__main__':
    loader = DataLoader()

    loader.shuffle()
    R, G, B = [0, 0, 0]
    for i in range(len(loader.images_urls)):
        if i % 100 == 0:
            print(i)
        image = loader.get_image(loader.images_urls[i])
        B += (np.sum(image[:, :, 0]) / (224 * 224))
        G += (np.sum(image[:, :, 1]) / (224 * 224))
        R += (np.sum(image[:, :, 2]) / (224 * 224))
    B /= len(loader.images_urls)
    G /= len(loader.images_urls)
    R /= len(loader.images_urls)
    print(B, G, R)

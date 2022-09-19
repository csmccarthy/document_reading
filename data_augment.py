from matplotlib import pyplot as plt
import numpy as np
from math import floor, ceil
import pickle
import cv2
import random
import gzip
from mnistHelper import mnistHelper

class DataAug:

    percent = None
    augfactor = None #[1, 2, 2, 2, 1, 1]
    data = None
    labels = None
    img_size = 28
    savefolder = "C:\\Users\\csmccarthy\\Documents\\pythonprojects\\augmented_mnist\\"
    
    def __init__(self, multipliers):
        """ Loads MNIST data, stores parameters. Augfactor layout: [res, rot, x, y, c, sp]  """

        self.augfactor = multipliers
        self.data, self.labels = mnistHelper.load_data(train=True, blanks = True)
        self.data = self.data.reshape([-1, self.img_size, self.img_size])
        self.data = list(self.data)
        self.labels = list(self.labels)
        self.percent = 100/len(self.data)


    def augment(self):
        """ Runs all augmentation functions on data """

        for factor in self.augfactor:
            if factor > 1:
                newlbls = []
                for i in range(len(self.labels)):
                    lbl = self.labels.pop(0)
                    for j in range(factor):
                        newlbls.append(lbl)
                    self.labels.extend(newlbls)
                    newlbls = []

        transform_array = [resize, rotate, xtrans, ytrans, contrast, saltnpepper]
        for i in range(len(transform_array)):
            if self.augfactor[i] != 0:
                transform_array[i]()
                    
        print('Done!')

    
    def contrast(self):

        augs = []
        for i in range(len(self.data)):
            if i % 100 == 0: print(f"Contrast variation is {int(i*self.percent)}% done")
            img = self.data.pop(0)
            for j in range(self.augfactor[4]):
                contrast = (random.random()*0.3)+0.2
                shifted = img-contrast
                floored = np.where(shifted >= 0, shifted, img)
                augs.append(floored)
            self.data.extend(augs)
            augs = []


    def xtrans(self, shiftrange=2):

        augs = []
        for i in range(len(self.data)):
            if i % 100 == 0: print(f"X variation is {int(i*self.percent)}% done")
            img = self.data.pop(0)
            for j in range(self.augfactor[2]):
                x_dis = random.randint(-shiftrange, shiftrange)
                fill = np.zeros([self.img_size, abs(x_dis)], dtype=np.float64)
                if x_dis < 0:
                    shifted = img[:, :x_dis]
                    shifted = np.concatenate((fill, shifted), axis=1)
                elif x_dis > 0:
                    shifted = img[:, x_dis:]
                    shifted = np.concatenate((shifted, fill), axis=1)
                else:
                    shifted = img
                augs.append(shifted)
            self.data.extend(augs)
            augs = []


    def ytrans(self, shiftrange=2):

        augs = []
        for i in range(len(self.data)):
            if i % 100 == 0: print(f"Y variation is {int(i*self.percent)}% done")
            img = self.data.pop(0)
            for j in range(self.augfactor[3]):
                y_dis = random.randint(-shiftrange, shiftrange)
                fill = np.zeros([abs(y_dis), self.img_size], dtype=np.float64)
                if y_dis < 0:
                    shifted = img[:y_dis, :]
                    shifted = np.concatenate((fill, shifted), axis=0)
                elif y_dis > 0:
                    shifted = img[y_dis:, :]
                    shifted = np.concatenate((shifted, fill), axis=0)
                else:
                    shifted = img
                augs.append(shifted)
            self.data.extend(augs)
            augs = []


    def saltnpepper(self, noise_ratio=0.025):
        
        augs = []
        squares = self.img_size*self.img_size
        noise = int(squares*noise_ratio)
        idx = np.concatenate((np.ones(noise), np.zeros(squares-noise)))
        for i in range(len(self.data)):
            if i % 100 == 0: print(f"Noise variation is {int(i*self.percent)}% done")
            img = self.data.pop(0)
            for j in range(self.augfactor[5]):
                np.random.shuffle(idx)
                choice = np.reshape(idx, (self.img_size,self.img_size))
                temp = np.where(choice == 1, 0, img)

                np.random.shuffle(idx)
                choice = np.reshape(idx, (self.img_size,self.img_size))
                temp = np.where(choice == 1, 1, temp)
                augs.append(temp)
            self.data.extend(augs)
            augs = []

            
    def rotate(self, maxangle=15):
        
        augs = []
        for i in range(len(self.data)):
            if i % 100 == 0: print(f"Rotational variation is {int(i*self.percent)}% done")
            img = self.data.pop(0)
            for i in range(self.augfactor[1]):
                angle = (random.random()*2*maxangle)-maxangle
                parent = np.zeros([40, 40], dtype=np.float64)
                parent[6:34, 6:34] = img
                rot = cv2.getRotationMatrix2D((20,20), angle, 1)
                parent = cv2.warpAffine(parent, rot, (40,40))
                temp = parent[6:34, 6:34]
                augs.append(temp)
            self.data.extend(augs)
            augs = []


    def resize(self):
        
        augs = []
        for i in range(len(self.data)):
            if i % 100 == 0: print(f"Size variation is {int(i*self.percent)}% done")
            img = self.data.pop(0)
            for i in range(self.augfactor[0]):
                ratio = (random.random()*0.2)+0.8
                temp = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
                padval = (self.img_size-(temp.shape[0]))/2
                pad = (floor(padval),ceil(padval))
                temp = np.pad(temp, (pad,pad), 'constant', constant_values=0)
                augs.append(temp)
            self.data.extend(augs)
            augs = []


    def save(self, version):
        
        with open(self.savefolder + "aug_mnist_" + version + ".pkl", 'wb') as f:
            pickle.dump(self.data, f)
        with open(self.savefolder + "aug_mnist_" + version + ".pkl", 'wb') as f:
            pickle.dump(self.labels, f)


    def preview(self, num):

        for pic in self.data[:num]:
            plt.imshow(pic)
            plt.show()

    

    

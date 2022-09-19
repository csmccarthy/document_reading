import cv2
import numpy as np
import sys
import os
from math import *
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model

class TemplateReader:
    """ Parent class for both types of template readers, stores KMeans and template matching functions """

    @staticmethod
    def fitClusters(sizes, points, tolerance = 15):
        """ Estimates how many clusters to use within a certain range """
        
        best = sys.maxsize # keeps running total of best inertia
        previous = sys.maxsize
        topnum = 0
        toplabels = []
        for n in sizes:
            if n <= len(points):  # skip if you try to fit n points to >n clusters
                kmeans = KMeans(n_clusters=n)
                kmeans = kmeans.fit(points)
                #  inertial improvement must pass a threshold, else clusters are arbitrarily split in two
                if kmeans.inertia_ < best and abs(previous - kmeans.inertia_) > tolerance: 
                    best = kmeans.inertia_ 
                    topnum = n
                    toplabels = kmeans.labels_
                previous = kmeans.inertia_
                
        return topnum, toplabels

    @staticmethod
    def trimPoints(ptprobs, threshold = 3):
        """ Takes in points of tl corner of a template, pops outliers (based on y-distance) and digit confidence
            Shape of ptprobs: [[(x,y), confidence, key], ...] """

        yavg = sum(pt[1] for pt in ptprobs[:, 0])/len(ptprobs) ### maybe change to median instead of average, to devalue outliers? ###
        ptprobs = np.asarray([pt for pt in ptprobs if abs(pt[0][1] - yavg) < threshold]) 
        
        return ptprobs


    @staticmethod
    def findMatch(frame, prefix, hasKey, templates, threshold = 0.7):
        """ Matches all digit templates above a certain threshold to a given image
            Returns matches in the form [ [(x,y), confidence, key (if hasKey)], ...] """
        
        ptprobs = []
        for template in templates: # loop through images to be matched -- digits, boxes, etc
            for i in range(4): # check through different variations on that single image, match them all
                filepath = prefix + template + f'-{i}.png'
                if os.path.isfile(filepath): # only match if the template has that many variations
                    num = cv2.imread(filepath, 0)
                    res = cv2.matchTemplate(frame, num, cv2.TM_CCOEFF_NORMED)
                    locs = np.where(res >= threshold) # discard poor matches
                    if hasKey:
                        for j in range(len(locs[0])):
                            ptprobs.append([(locs[1][j], locs[0][j]), res[locs[0][j]][locs[1][j]], template])
                    else:
                        for j in range(len(locs[0])):
                            ptprobs.append([(locs[1][j], locs[0][j]), res[locs[0][j]][locs[1][j]]])
        return np.asarray(ptprobs)

    @staticmethod
    def getSequence(digit_probs, points, cluster_number, cluster_labels, has_key):
        """ Takes in clusters and the keys they contain, returns the highest confidence key per cluster
            Returns [ [(x,y), key (if hasKey)], ...] """
        
        seq = []
        top_dict = {}
        for cluster in range(cluster_number):
            top_dict[cluster] = [None, 0] # for a given cluster, store the index of the top match and its confidence
        for ix, cnum in enumerate(cluster_labels): # iterate through clusters, set top_dict with best match per cluster
            if digit_probs[ix][0] > top_dict[cnum][1]:
                top_dict[cnum] = [ix, digit_probs[ix][0]]
        if has_key: # add only the best match per cluster to seq, and its key if needed
            for cluster in range(cluster_number):
                topix = top_dict[cluster][0]
                seq.append([points[topix], digit_probs[topix][1]])
        else:
            for cluster in range(cluster_number):
                topix = top_dict[cluster][0]
                seq.append(points[topix])
                
        return seq
    


class DigitReader(TemplateReader):
    """ Stores data read from fields, computes number sequences from kmeans clusters """

    studyprefix = cv2.imread('C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/study_prefix.png', 0)
    info_dict = {}
    digit_dict = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
                  '6': '6', '7': '7', '8': '8', '9': '9', '10': '.', '11': '<', '12': '%'}
    field_dict = {'eu': (slice(1256, 1280), slice(890, 970)), 'ppc': (slice(1345, 1371), slice(550, 610)),
                  'vol': (slice(315, 336), slice(735, 800)), 'dev': (slice(314, 336), slice(299, 330)),
                  'study': (slice(1185, 1210), slice(62, 137)), 'limit': (slice(1255, 1280), slice(1080, 1140))}

    def readTemplate(self, page_scan, template_num):
        """ Reads template specific numbers, keeps track of characters present/length of digits """
        
        if template_num == 0:
            res = cv2.matchTemplate(page_scan, self.studyprefix, cv2.TM_CCOEFF_NORMED)
            _,_,_,max_loc = cv2.minMaxLoc(res)
            
            self.readNumber(templates_present = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                            page_img = page_scan, page_key = 'study', number_len = [7], shift=max_loc[0])
            self.readNumber(templates_present = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
                            page_img = page_scan, page_key = 'eu', number_len = range(3,9))
            self.readNumber(templates_present = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '12'],
                            page_img = page_scan, page_key = 'ppc', number_len = range(3,5))
            self.readNumber(templates_present = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                            page_img = page_scan, page_key = 'limit', number_len = range(2,5))

        elif template_num == 1:
            self.readNumber(templates_present = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                            page_img = page_scan, page_key = 'vol', number_len = range(3,7))
            self.readNumber(templates_present = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                            page_img = page_scan, page_key = 'dev', number_len = range(1,3))
            
            newvol = float(self.info_dict['vol'])/float(self.info_dict['dev'])
            self.info_dict['vol'] = str(round(newvol, 1))

        tmp = self.info_dict
        self.info_dict = {}
        
        return tmp

    def readNumber(self, page_key, templates_present, number_len, page_img, shift = None):
        """ Takes page img and parameters for the data field, returns string of the number """
        
        insert = self.padInput(page_img, page_key, start = shift)
        ptprobs = self.findMatch(frame = insert, templates = templates_present, hasKey = True,
                                 prefix='C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/study')
        ptprobs = self.trimPoints(ptprobs)
        topnum, toplabels = self.fitClusters(sizes=number_len, points=list(ptprobs[:, 0]), tolerance=15)

        # INCLUDE FOR CLUSTER VISUALIZATION
        newfield = cv2.cvtColor(insert, cv2.COLOR_GRAY2RGB)
        for idx, match in enumerate(ptprobs):
            r = (50 + int(toplabels[idx]) * 70) % 255
            g = (100 + int(toplabels[idx]) * 25) % 255
            b = (150 + int(toplabels[idx]) * 100) % 255
            cv2.circle(newfield, match[0], 1, (r, g, b), -1)
        cv2.imwrite('C:/Users/csmccarthy/Documents/pythonprojects/pagereading/cluster_match.png', newfield)
        plt.imshow(newfield)
        plt.show()
        
        numberseq = self.getSequence(digit_probs = ptprobs[:, 1:], points = ptprobs[:, 0],
                                     cluster_number = topnum, cluster_labels = toplabels, has_key= True)
        plaintext = self.seq2text(numberseq)
        self.info_dict[page_key] = plaintext

    def padInput(self, frame, key, insert_w = 80, insert_h = 30, start = None):
        """ Surrounds image of interest with whitespace for ease of template matching """

        rect = self.field_dict[key] # field_dict stores data for slicing page img into digit imgs corresponding to key
        if key == 'study': # study number can appear in a horizontal range, shift by start to account for this
            field = frame[rect[0], start+rect[1].start:start+rect[1].stop] 
        else:
            field = frame[rect[0], rect[1]]
        hpad = (insert_h - field.shape[0])/2
        topbot = (floor(hpad), ceil(hpad)) # split vertical padding between the top and bottom of the data
        leftright = (0, insert_w - field.shape[1]) # add all horizontal padding to the right side
        padded = np.pad(field, (topbot, leftright), 'constant', constant_values=255) # fill in with whitespace
        return padded

    def seq2text(self, sequence):
        """ Takes unorganized sequence of numbers, sorts by position and returns string """
        
        sequence = sorted(sequence, key = lambda x: x[0][0])
        plaintext = ''
        for num in sequence:
            plaintext += self.digit_dict[num[1]]
        return plaintext



class BoxReader(TemplateReader):
    
    prep_box = cv2.imread('C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/box.png', 0)
    css_box = cv2.imread('C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/box2-0.png', 0)
    prep_h, prep_w = prep_box.shape
    css_h, css_w = css_box.shape
    prep_margin = 4
    css_margin = 5

    prep_checks = {'immerseinc': [slice(65,100), slice(325,365)], 'immerseroom': [slice(130,165), slice(325,365)],
                'flushed': [slice(195,225), slice(325,365)], 'addcomp': [slice(235,270), slice(370,410)],
                'nocomp': [slice(280,310), slice(370,410)]}

    css_checks = {'css': [slice(305, 380), slice(710, 915)], 'rev': [slice(305, 380), slice(940, 1065)]}
    
    spec_checks = {}
    info_dict = {}
    model = None

    def __init__(self):
        self.model = load_model('/tmp/new_cnn_augmented3')
        

    def findCentroid(self, img, length=28):
        """ Estimates y-centroid of an digit image using real values """
        """ Input image must use black as negative space, with white strokes """
        
        ycenter, xcenter = 0, 0
        ylinecount, xlinecount = 0, 0

        for i in range(length):
            ysliced = img[:,i]
            ylinesum = sum(ysliced)
            if ylinesum > 50:
                ylinecount += 1
                ixs = np.arange(length)
                ycenter += np.dot(ysliced, ixs) / ylinesum

            xsliced = img[i, :]
            xlinesum = sum(xsliced)
            if xlinesum > 50:
                xlinecount += 1
                ixs = np.arange(length)
                xcenter += np.dot(xsliced, ixs) / xlinesum
        ycenter /= ylinecount
        xcenter /= xlinecount
        
        return int(xcenter), int(ycenter)
    

    def findMedian(self, img, length=28):
        """ Estimates median point of an image using binary masking """
        """ Input image must use black as negative space, with white strokes """

        mask = (img >= 20).astype(np.int)
        yfirst, xfirst = None, None
        ylast, xlast = None, None

        for i in range(length):
            ysliced = mask[i,:]
            if (1 in ysliced) and yfirst is None:
                yfirst = i
            elif yfirst != None and (1 in ysliced):
                ylast = i
                
            xsliced = mask[:,i]
            if (1 in xsliced) and xfirst is None:
                xfirst = i
            elif xfirst is not None and (1 in xsliced):
                xlast = i
        ycenter = (yfirst+ylast)/2
        xcenter = (xfirst+xlast)/2
        
        return (int(xcenter), int(ycenter))
    

    def centerDigit(self, img, length=28):
        """ Takes in an image of a digit and centers it in the frame, using an average of the median and centroid """
        
        xmedian, ymedian = self.findMedian(img)
        xcentroid, ycentroid = self.findCentroid(img)
        xcenter = (xmedian+xcentroid)//2
        ycenter = (ymedian+ycentroid)//2
        
        yoffset = ycenter-14
        yshift = abs(yoffset)
        yblank = np.zeros((yshift, 28))
        if yoffset > 0:
            img = np.concatenate((img[yshift:, :], yblank), axis=0)
        elif yoffset < 0:
            img = np.concatenate((yblank, img[:-yshift, :]), axis=0)

        xoffset = xcenter-14
        xshift = abs(xoffset)
        xblank = np.zeros((28, xshift))
        if xoffset > 0:
            img = np.concatenate((img[:, xshift:], xblank), axis=1)
        elif xoffset < 0:
            img = np.concatenate((xblank, img[:, :-xshift]), axis=1)
            
        return img
    

    def readPrepBoxes(self, img):
        plt.imshow(img, cmap='gray')
        plt.show()
        for key in self.bet_checks.keys():
            val = self.bet_checks[key]
            field = img[val[0], val[1]]
            
            res = cv2.matchTemplate(field, self.prep_box, cv2.TM_CCOEFF_NORMED)
            _, _, _, loc = cv2.minMaxLoc(res)
            tl = (loc[0]+self.prep_margin, loc[1]+self.prep_margin)
            br = (loc[0]+self.prep_h-self.prep_margin, loc[1]+self.prep_w-self.prep_margin)
            
            checkarea = field[tl[1]:br[1], tl[0]:br[0]]
            plt.imshow(checkarea, cmap='gray')
            plt.show()
            checkarea = np.where(checkarea < 210, 1, 0)
            plt.imshow(checkarea, cmap='gray')
            plt.show()
            checksum = np.sum(checkarea)
            print(f"Checksum = {checksum}")
            if checksum > 5:
                self.info_dict[key] = True
            else:
                self.info_dict[key] = False

        tmp = self.info_dict
        self.info_dict = {}
        
        return tmp

    def readCSSBoxes(self, img):
        plt.imshow(img, cmap='gray')
        plt.show()
        for key in self.css_checks.keys():
            val = self.css_checks[key]
##            field = img[val[0], val[1]]
            if key == 'css':
                sizes = [4]
                field = img[:, :300]
            else:
                sizes = [2]
                field = img[:, 300:]

            ptprobs = self.findMatch(frame = field, templates = ['box2'], hasKey = False,
                                     prefix='C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/')

            
            ptprobs = self.trimPoints(ptprobs)
            topnum, toplabels = self.fitClusters(sizes = sizes, points = list(ptprobs[:, 0]))

            show = cv2.cvtColor(field, cv2.COLOR_GRAY2RGB)

            for idx, pt in enumerate(ptprobs[:, 0]):
                r = 50+int(toplabels[idx])*70
                g = (100+int(toplabels[idx])*25)%255
                b = (150+int(toplabels[idx])*100)%255
                cv2.circle(show, pt, 2, (r, g, b), -1)
            cv2.imwrite('C:/Users/csmccarthy/Documents/pythonprojects/pagereading/cluster_match.png', show)
            plt.imshow(show, cmap='gray')
            plt.show()

            ptseq = self.getSequence(digit_probs = ptprobs[:, 1:], points = ptprobs[:, 0],
                                     cluster_number = topnum, cluster_labels = toplabels, has_key=False)
            ptseq = sorted(ptseq, key = lambda x: x[0])

            seq = ''
            for pt in ptseq:
                hslice = slice(pt[0]+self.css_margin, pt[0]+self.css_h-self.css_margin)
                wslice = slice(pt[1]+self.css_margin, pt[1]+self.css_w-self.css_margin)
                sliced = field[wslice, hslice]
                sliced = cv2.resize(sliced, (26, 26))
                sliced = np.pad(sliced, (1,1), 'constant', constant_values=255)
                inverted = (255-sliced)
                inverted = self.centerDigit(inverted)
                normal = inverted / 255
                plt.imshow(normal, cmap='gray')
                plt.show()
                normal = normal.reshape([-1, 28, 28, 1])
                seq += str(np.argmax(self.model.predict(normal)))

            print(seq)
            
            self.info_dict[key] = seq

        tmp = self.info_dict
        self.info_dict = {}
        
        return tmp
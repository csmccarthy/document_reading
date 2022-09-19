import cv2
import numpy as np
from matplotlib import pyplot as plt

class PageMatcher:
    """ Stores template info, along with current page data """

    templates = [['C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/restemp1_blank.png',
                  'C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/restemp1_blank_L.png',
                  'C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/restemp1_blank_R.png'],
                 ['C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/restemp2_blank.png',
                  'C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/restemp2_blank_L.png',
                  'C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/restemp2_blank_R.png'],
                 ['C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/preptemp1_blank_digital.png',
                  'C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/preptemp1_blank_digital_L.png',
                  'C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/preptemp1_blank_digital_R.png'],
                 ['C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/betprep-2.png',
                  'C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/betprep-2_L.png',
                  'C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/betprep-2_R.png',
                  'C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/betprep-2_Lrot.png',
                  'C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/betprep-2_Rrot.png',]]
    
        
    good_matches = 30
    page_scan = []
    alt_num = 5
    hw = []
    kp = []
    des = []
    template_kp = []
    template_des = []
    orb = None
    orb2 = None
    matcher = None
    matcher2 = None
    
    def __init__(self):
        self.orb = cv2.ORB_create(2000)
        self.orb2 = cv2.ORB_create(1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        self.matcher2 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        for i, template in enumerate(self.templates):
            self.hw.append([])
            self.template_kp.append([])
            self.template_des.append([])
            for alt in template:
                template_img = cv2.imread(alt)
                self.hw[i].append(template_img.shape)
                tkp, tdes = self.orb.detectAndCompute(template_img, None)
                self.template_kp[i].append(tkp)
                self.template_des[i].append(tdes)


# hw - [[[t1-h, t1-w], [t1L-h, t1L-w], [t1R-h, t1R-w]],   (stores template height/width)
#      [[t2-h, t2-w], [t2L-h, t2L-w], [t2R-h, t2R-w]]]
# kp  - [[t1, t1L, t1R], [t2, t2L, t2R]]                  (stores template keypoints)
# des - [[t1, t1L, t1R], [t2, t2L, t2R]]                  (stores template descriptions)


    def detectMatch(self, page_name):
        """ Takes in descriptions and checks them against all previously computed template descriptions, returns good matches """
        
        self.page_scan = cv2.imread(page_name, 0)
##        plt.imshow(self.page_scan, cmap='gray') # uncomment if you want to see the individual pages
##        plt.show()
        self.kp, self.des = self.orb.detectAndCompute(self.page_scan, None)
        
        #code this to actually leverage the different templates!!

        template_matches = []
        for template_num, tdes in enumerate(self.template_des):
            if template_num == 2:
                matchfound, matches, matchdist = self.sectionMatch(template_num, (3,5), (0,1), numMatches=30)
                if matchfound:
                    template_matches.append([matches, 2, matchdist])
            elif template_num == 3:
                matchfound, matches, matchdist = self.sectionMatch(template_num, (2,12), (1,2), numMatches=3)
                if matchfound:
                    template_matches.append([matches, 3, matchdist])
            else:
                matchfound, matches, matchdist = self.pageMatch(template_num, tdes)
                if matchfound:
                    template_matches.append([matches, template_num, matchdist])
        
        if len(template_matches) == 0: # return whether a match was found, its template number, its matches, and distance
            return (False, -1, None, None)
        else:
            template_matches = sorted(template_matches, key = lambda x: x[2]) # return the matchlist with the lowest distance
            return (True, template_matches[0][1], template_matches[0][0], template_matches[0][2])


    def sectionMatch(self, template_num, yfrac, xfrac, numMatches=18):
        ychunk = self.page_scan.shape[0]//yfrac[1]
        xchunk = self.page_scan.shape[1]//xfrac[1]

        #slice page into specified section
        if template_num == 2:
            template_img = cv2.imread('C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/preptemp1_filled.png', 0) ## make this variable!
        else:
            template_img = cv2.imread('C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/betprep-2.png', 0)

        yslice = slice(ychunk*yfrac[0], ychunk*(yfrac[0]+1))
        xslice = slice(xchunk*xfrac[0], xchunk*(xfrac[0]+1))
        section = self.page_scan[yslice, xslice]
        template_section = template_img[yslice, xslice]
        plt.imshow(section)
        plt.show()
        plt.imshow(template_section)
        plt.show()

        # need to detect new set of keypoints (since they all have to be within the section), cant just use the default template_kp
        kp, des = self.orb.detectAndCompute(section, None)
        tkp, tdes = self.orb.detectAndCompute(template_section, None)

        tmpmatch = self.matcher.match(des, tdes)
        tmpmatch = sorted(tmpmatch, key = lambda x: x.distance)

        matchimg = cv2.drawMatches(section, kp, template_section, tkp, tmpmatch[:30], None)
        plt.imshow(matchimg)
        plt.show()

        tmpmatch = tmpmatch[:self.good_matches]
        matchdist = sum(m.distance for m in tmpmatch)
        tmpmatch = [m for m in tmpmatch if m.distance < 25]
        if len(tmpmatch) != 0: print("template " + str(template_num) + " avg dist: " + str(matchdist/len(tmpmatch)))
        print('number of matches: ' + str(len(tmpmatch)))
        
        ### SCALE NUMBER OF REQUIRED MATCHES WITH THE SIZE OF THE SECTION BEING MATCHED ########################
        
        if len(tmpmatch) > numMatches:
            return (True, tmpmatch, matchdist/len(tmpmatch))
        else:
            return (False, None, None)


    def pageMatch(self, template_num, des):
        
        tmpmatch = self.matcher.match(self.des, des[0])
        tmpmatch = sorted(tmpmatch, key = lambda x: x.distance)
        template_img = cv2.imread(self.templates[template_num][0], 0)
        
        tmpmatch = tmpmatch[:self.good_matches] # trim to only take the best matches per template
        
        matchdist = 0
        close = True
        for match in tmpmatch: # if any one match is too distant, do not include that template in the decision
            matchdist += match.distance
            if match.distance > 12:
                close = False

        tmp = [m for m in tmpmatch if m.distance < 12]
        if len(tmp) != 0: print("template " + str(template_num) + " avg dist: " + str(matchdist/len(tmp)))
        print('number of matches: ' + str(len(tmp)))
        
        if close:
            return (True, tmpmatch, matchdist/len(tmpmatch))
        else:
            return (False, None, None)


    def sectionAdjust(self, matches, template_num, numer, denom):
        """ Takes the number key of the page to match, matches a section of it, and returns a perspective transformed version to match the template """

        fraction = self.page_scan.shape[0]//denom

##        if template_num == 2:
##            template_img = cv2.imread('C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/preptemp1_filled.png', 0)
##        else:
##            template_img = cv2.imread('C:/Users/csmccarthy/Documents/pythonprojects/betscans/templates/betprep-2.png', 0)
        section = self.page_scan[fraction*numer:fraction*(numer+1), self.page_scan.shape[1]//2:]
##        template_section = template_img[fraction*numer:fraction*(numer+1), :]

        kp, des = self.orb.detectAndCompute(section, None)
        tkp = []
        tdes = []
        for i, alt in enumerate(self.templates[template_num]):
            template_img = cv2.imread(alt, 0)
            sliced = template_img[fraction*numer:fraction*(numer+1), template_img.shape[1]//2:]
            tempkp, tempdes = self.orb.detectAndCompute(sliced, None)
            
            tkp.append([])
            tkp[i] = tempkp
            tdes.append([])
            tdes[i] = tempdes
        
        pts1 = np.zeros([self.alt_num, len(matches), 2])
        pts2 = np.zeros([self.alt_num, len(matches), 2])

        altmatches = [] ### ABSTRACT THIS INTO ANOTHER METHOD ######################################################
        altmatches.append(matches)
        for i in range(1, self.alt_num):
            tmp = self.matcher.match(des, tdes[i])
            tmp = sorted(tmp, key = lambda x:x.distance)
            tmp = tmp[:len(matches)]
            altmatches.append(tmp)
        
        for i, matchlist in enumerate(altmatches):
            for j, match in enumerate(matchlist):
                pts1[i,j,:] = kp[match.queryIdx].pt
                pts2[i,j,:] = (tkp[i])[match.trainIdx].pt

        avgtrans, mask = cv2.findHomography(pts1[0], pts2[0], cv2.RANSAC)
        for i in range(1, self.alt_num):
            trans, mask = cv2.findHomography(pts1[i], pts2[i], cv2.RANSAC)
            tmp = np.add(avgtrans, trans)
            avgtrans = np.divide(tmp, 2)
            
        scanreg = cv2.warpPerspective(section, avgtrans, (self.page_scan.shape[1]//2, fraction))

        return scanreg


    def pageAdjust(self, matches, template_num):
        """ Takes the number key of the page to match and returns a perspective transformed version to match the template """
        
        pts1 = np.zeros([self.alt_num, self.good_matches, 2])
        pts2 = np.zeros([self.alt_num, self.good_matches, 2])

        # newmatches = [[list of t# matches], [list of t#L matches], [list of t#R matches]]
        altmatches = []
        altmatches.append(matches)
        for i in range(1, len(self.templates[template_num])):
            tmp = self.matcher.match(self.des, self.template_des[template_num][i])
            tmp = sorted(tmp, key = lambda x:x.distance)
            tmp = tmp[:self.good_matches]
            altmatches.append([m for m in tmp if m.distance < 10])
        
        for i, matchlist in enumerate(altmatches):
            for j, match in enumerate(matchlist):
                pts1[i,j,:] = self.kp[match.queryIdx].pt
                pts2[i,j,:] = (self.template_kp[template_num][i])[match.trainIdx].pt

        # compute multiple homography matrices for the same template (but with slight
        # differences in the position/rotation between templates) to hopefully make
        # template alignment more precise

        avgtrans, mask = cv2.findHomography(pts1[0], pts2[0], cv2.RANSAC)
        for i in range(1, len(self.templates[template_num])):
            trans, mask = cv2.findHomography(pts1[i], pts2[i], cv2.RANSAC)
            tmp = np.add(avgtrans, trans)
            avgtrans = np.divide(tmp, 2)

        #apply transform and return
        scanreg = cv2.warpPerspective(self.page_scan, avgtrans, (self.hw[template_num][2][1], self.hw[template_num][2][0]))   #(width, then height)
        return scanreg
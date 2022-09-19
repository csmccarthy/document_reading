from matplotlib import pyplot as plt
from PageMatcher import PageMatcher
from PageReader import DigitReader
from PageReader import BoxReader

def getPages(folder = 'C:/Users/csmccarthy/Documents/pythonprojects/newbetscans/'):
    """ Adds .png files to an array to be matched against page templates """
    
    with open(folder + 'pages.txt', 'r') as f:
        dump = f.readlines()
    pages = []
    for l in dump:
        # grab [:-1] to strip newline character
        pages.append(folder + l[:-1])
    return pages


def parsePages(pages):
    """ Generator function """
    """ Returns the filenames for all files with the same timestamp (from the same report) """
    
    chunk = []
    timestamp = pages[0][:80]
    for page in pages:
        if page == pages[-1]:
            chunk.append(page)
            yield chunk
        elif page[:80] == timestamp:
            chunk.append(page)
        else:
            yield chunk
            chunk = []
            chunk.append(page)
            timestamp = page[:80]
        

def saveInfo(info_dict):
    """ Checks to see that all the necessary info for one report has been read, writes it to a txt """

    if len(info_dict.keys()) == 13:
        with open('C:/Users/csmccarthy/Documents/pythonprojects/tobetyped/' + str(info_dict['study']) + '.txt', 'w') as f:
            for n, key in enumerate(info_dict.keys()):
                if n == 12:
                    f.write(key + ',' + str(info_dict[key]))
                else:
                    f.write(key + ',' + str(info_dict[key]) + ';')
        print("Save Successful")  
        return True
    else:
        print("Save Error")
        return False


page_matcher = PageMatcher()
dreader = DigitReader()
breader = BoxReader()

skip = False
maxskip = 10
skipcount = 0

info_dict = {}
match_dict = {}

pagenames = getPages()
# pagenames = ['C:\\Users\\csmccarthy\\Documents\\pythonprojects\\lal prep\\S19050712330-7.png',
#              'C:\\Users\\csmccarthy\\Documents\\pythonprojects\\lal prep\\S19050712330-1.png',
#              'C:\\Users\\csmccarthy\\Documents\\pythonprojects\\lal prep\\S19050712330-2.png',
#              'C:\\Users\\csmccarthy\\Documents\\pythonprojects\\lal prep\\S19050712330-3.png',
#              'C:\\Users\\csmccarthy\\Documents\\pythonprojects\\lal prep\\S19050712330-4.png',
#              'C:\\Users\\csmccarthy\\Documents\\pythonprojects\\lal prep\\S19050712330-5.png',
#              'C:\\Users\\csmccarthy\\Documents\\pythonprojects\\lal prep\\S19050712330-6.png',
#              'C:\\Users\\csmccarthy\\Documents\\pythonprojects\\lal prep\\S19050712330-7.png']

for pagechunk in parsePages(pagenames):
    for page in pagechunk:
        # for page in pagenames:
        matchfound, matchnum, matches, matchdist = page_matcher.detectMatch(page)

        if not matchfound:
            print("Match not found.")

        if matchfound and skip:
            print(f"Matched to template {matchnum+1}, Skipping report - ({skipcount})")
            if skipcount == (maxskip): skip = False
            skipcount += 1

        elif matchfound:
            print("Matched to template " + str(matchnum+1) + ".")


            #you should maybe route the keypoints and descriptors through here?

            if matchnum == 0 or matchnum == 1:
                if matchnum not in match_dict.keys():
                    match_dict[matchnum] = matchdist
                    print(f"match distance is {matchdist}")
                    scanreg = page_matcher.pageAdjust(matches, matchnum)

                    info_dict.update(dreader.readTemplate(scanreg, matchnum))
                    print(info_dict)

                    plt.imshow(scanreg, cmap='gray')
                    plt.show()
                else:
                    print("Already Matched")

            elif matchnum == 2:
                if matchnum not in match_dict.keys():
                    match_dict[matchnum] = matchdist
                    print(f"match distance is {matchdist}")
                    scanreg = page_matcher.sectionAdjust(matches, matchnum, 3, 5)

                    info_dict.update(breader.readPrepBoxes(scanreg))
                    print(info_dict)

                    plt.imshow(scanreg, cmap='gray')
                    plt.show()
                else:
                    print("Already Matched")
            elif matchnum == 3:
                ##            if matchnum not in match_dict.keys():
                match_dict[matchnum] = matchdist
                print(f"match distance is {matchdist}")
                scanreg = page_matcher.sectionAdjust(matches, matchnum, 2, 12)

                info_dict.update(breader.readCSSBoxes(scanreg))
                print(info_dict)

                plt.imshow(scanreg, cmap='gray')
                plt.show()
##            else:
##                print("Already Matched")
match_dict = {}
print("Saving.....")
if saveInfo(info_dict):
    info_dict = {}
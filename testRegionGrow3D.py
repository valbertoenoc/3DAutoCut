import scipy.io as sio
import numpy as np

import re
seeds = ()
seedI = []
seedJ = []
seedZ = []

def startRG2D(image, seedsIndex, similarity):

    result = np.zeros(image.shape, dtype=image.dtype)
    pixelAvg = np.average(image[seedsIndex])

    #if no seeds, return
    if len(seedsIndex) == 0:
        print("No seeds created.")
        return

    #change pixels of result image to seeds
    result[seedsIndex] = image[seedsIndex]

    #convert seedindex tuple into list of seeds
    seeds = [(seedsIndex[0][i], seedsIndex[1][i]) for i in range(len(seedsIndex[0]))]
    print(len(seeds))

    #get image dimensions
    width = image.shape[1]
    height = image.shape[0]

    checked_seeds = np.zeros(image.shape, dtype='uint8')
    checked_seeds[seedsIndex] = 1

    sd = 0
    while sd < len(seeds):

        i = seeds[sd][0]
        j = seeds[sd][1]

        # neighbor loop
        for n in range(-1, 2, 2):

            # is seed inside image
            if (i + n > 0 and i + n < height) and (j + n > 0 and j + n < width):

                pix_n_index_i = (i + n, j)
                pix_n_index_j = (i, j + n)

                pixel_i = image[pix_n_index_i]

                if checked_seeds[pix_n_index_i] == 0 and abs(pixel_i - pixelAvg) < abs(similarity*pixelAvg):
                    result[pix_n_index_i] = image[pix_n_index_i]
                    seeds.append(pix_n_index_i)
                    checked_seeds[pix_n_index_i] = 1

                pixel_j = image[pix_n_index_j]

                if checked_seeds[pix_n_index_j] == 0 and abs(pixel_j - pixelAvg) < abs(similarity*pixelAvg):
                    result[pix_n_index_j] = image[pix_n_index_j]
                    seeds.append(pix_n_index_j)
                    checked_seeds[pix_n_index_j] = 1

                # cv2.imshow('growing', result)
                # cv2.waitKey(1)

        sd += 1

    return result

def startRG3D(image, seedsIndex,window_size, similarity):

    result = np.zeros(image.shape, dtype=image.dtype)
    pixelAvg = np.average(image[seedsIndex])

    #if no seeds, return
    if len(seedsIndex) == 0:
        print("No seeds created.")
        return

    #change pixels of result image to seeds
    result[seedsIndex] = image[seedsIndex]

    #convert seedindex tuple into list of seeds
    seeds = [(seedsIndex[0][i], seedsIndex[1][i], seedsIndex[2][i]) for i in range(len(seedsIndex[0]))]
   # print(len(seeds))

    #get image dimensions
    width = image.shape[2]
    height = image.shape[1]
    depth = image.shape[0]

    # checked seed dict
    checked_seeds = np.zeros(image.shape, dtype='uint8')
    checked_seeds[seedsIndex] = 1

    sd = 0
    ws = (window_size - 1) // 2

    for n in range(-ws, 1 + ws, 1):
        if (n == 0): continue

    while sd < len(seeds):

        z = seeds[sd][0]
        i = seeds[sd][1]
        j = seeds[sd][2]

        # neighbor loop
        for n in range(-ws, 1 + ws, 1):
            if (n == 0): continue

            # is seed inside image
            if (n==0): continue

            if (z + n > 0 and z + n < depth) and (i + n > 0 and i + n < height) and (j + n > 0 and j + n < width):

                pix_n_index_i = (z, i + n, j)
                pix_n_index_j = (z, i, j + n)
                pix_n_index_z = (z + n, i, j)

                pixel_i = image[pix_n_index_i]

                if checked_seeds[pix_n_index_i] == 0 and abs(pixel_i - pixelAvg) < abs(similarity*pixelAvg):
                    result[pix_n_index_i] = image[pix_n_index_i]
                    seeds.append(pix_n_index_i)
                    checked_seeds[pix_n_index_i] = 1

                pixel_j = image[pix_n_index_j]

                if checked_seeds[pix_n_index_j] == 0 and abs(pixel_j - pixelAvg) < abs(similarity*pixelAvg):
                    result[pix_n_index_j] = image[pix_n_index_j]
                    seeds.append(pix_n_index_j)
                    checked_seeds[pix_n_index_j] = 1

                pixel_z = image[pix_n_index_z]

                if checked_seeds[pix_n_index_z] == 0 and abs(pixel_z - pixelAvg) < abs(similarity * pixelAvg):
                    result[pix_n_index_z] = image[pix_n_index_z]
                    seeds.append(pix_n_index_z)
                    checked_seeds[pix_n_index_z] = 1

                # cv2.imshow('growing', result)
                # cv2.waitKey(1)

        sd += 1

    return result


def postProcessing( exam):
    from scipy import ndimage
    from skimage import measure


    auxArray = np.copy(exam)
    lung = ndimage.binary_opening(auxArray).astype(np.int)
    blobs_labels = measure.label(lung, background=0)
    regions = measure.regionprops(blobs_labels)
    regions.sort(key=lambda x: x.area, reverse=True)

    to_keep = [r for r in regions if ((r.area > 1000))]

    lung = np.zeros((auxArray.shape))
    for num in range(0, to_keep.__len__()):
        lung[blobs_labels == to_keep[num].label] = 255

    lung = ndimage.binary_fill_holes(lung).astype(np.int)

    return 255*lung.astype("uint8")

from matplotlib import pyplot as plt

ground_truth = sio.loadmat("data/IgroundTruth.mat")
background = sio.loadmat("data/Ibackground.mat")
ground_truth = ground_truth['Igt']
#background = background['Iesfera']
pathInput="data/Icubo.mat"
#pathOutput="data/IesferaRG3D_noise.mat"
IReadData = sio.loadmat(pathInput)

samples = [i for i in IReadData.keys() if i.startswith('I') and i[-1].isdigit()]
Ori = [i for i in IReadData.keys() if i.startswith('I') and i[-1].isalpha() and not any(j.isdigit() for j in i)]


GC3Dresult={}

nIteration=10
threshold=0.2

import timeit

timewin3=[]
timewin5=[]
timewin7=[]
timewin11=[]

for i in xrange(0,len(samples)):

    IOriData = IReadData[samples[i]]
    IProcessData = np.copy(IOriData)

    noise = re.findall(r'\d+', samples[i])[0]


    print  'segmentation ' + samples[i] + '  ' + str(100*i/len(samples)) + '%'


    print str(i)+ ' Window 3'
    seeds=np.where(ground_truth)
    start = timeit.default_timer()
    segmentation=startRG3D(np.copy(IProcessData),seeds,3,threshold)
    stop = timeit.default_timer()
    timewin3.append(stop - start)

   # segmentation2=postProcessing(segmentation)
   # GC3Dresult[Ori[0] + 'RG3D' + noise + 'win3'] = segmentation2.astype('uint8')

    print str(i)+ ' Window 5'
    seeds=np.where(ground_truth)
    start = timeit.default_timer()
    segmentation=startRG3D(np.copy(IProcessData),seeds,5,threshold)
    stop = timeit.default_timer()
    timewin5.append(stop - start)
    #segmentation2=postProcessing(segmentation)

    #GC3Dresult[Ori[0] + 'RG3D' + noise + 'win5'] = segmentation2.astype('uint8')


    print str(i)+ ' Window 7'
    seeds=np.where(ground_truth)
    start = timeit.default_timer()
    segmentation=startRG3D(np.copy(IProcessData),seeds,7,threshold)
    stop = timeit.default_timer()
    timewin7.append(stop - start)
   # segmentation2=postProcessing(segmentation)

    #GC3Dresult[Ori[0] + 'RG3D' + noise + 'win7'] = segmentation2.astype('uint8')

    print str(i)+ ' Window 11'
    seeds=np.where(ground_truth)
    start = timeit.default_timer()

    segmentation=startRG3D(np.copy(IProcessData),seeds,11,threshold)
    stop = timeit.default_timer()
    timewin11.append(stop - start)
    #segmentation2=postProcessing(segmentation)

   # GC3Dresult[Ori[0] + 'RG3D' + noise + 'win11'] = segmentation2.astype('uint8')

    print '-'*50

#dice = computeDiceSimilarity3D(segmentation, ground_truth.astype("uint8"))
#print ("Dice ", dice)
print("terminou")

#sio.savemat(pathOutput,GC3Dresult)



#ct_metrics.computeDiceSimilarity3D()


import scipy.io as sio
import numpy as np
from scipy.ndimage.filters import gaussian_filter


from growcut3D import _growcut3D_cy
import cv2
import re

from ctprocessing import ct_metrics
from matplotlib import pyplot as plt
import timeit

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

ground_truth = sio.loadmat("data/IgroundTruth.mat")
IProcessData = sio.loadmat("data/Icubo.mat")
IProcessData = IProcessData['Icubo']
background = sio.loadmat("data/Ibackground.mat")
ground_truth = ground_truth['Igt']
#background = background['Iesfera']
pathInput="data/Icubo.mat"
#pathOutput="data/IesferaGC3D_noise.mat"
IReadData = sio.loadmat(pathInput)

samples = [i for i in IReadData.keys() if i.startswith('I') and i[-1].isdigit()]
Ori = [i for i in IReadData.keys() if i.startswith('I') and i[-1].isalpha() and not any(j.isdigit() for j in i)]


GC3Dresult={}

nIteration=10
timewin3=[]
timewin5=[]
timewin7=[]
timewin11=[]
for i in xrange(0,len(samples)):

    IOriData = IReadData[samples[i]]
    IProcessData = np.copy(IOriData)

    index = 1
    spread = 0.1
    # foreground_indices  np.nonzero(ground_truth.flatten())[0]
    # background_indices = np.nonzero(~(ground_truth.flatten()))[0]

    label = np.zeros(ground_truth.shape, dtype=np.float64)

    # random_foreground_indices = foreground_indices[np.random.randint(0, len(foreground_indices), spread * len(foreground_indices))]
    # random_background_indices = background_indices[np.random.randint(0, len(background_indices), spread * len(background_indices))]

    label[ground_truth > 0] = 1
    #label[background > 0] = -1

    strength = np.zeros_like(IProcessData, dtype=np.float64)
    strength[label == 1] = 1.0
    #strength[label == -1] = 1.0
    strength[label == 0] = 0.98

    labelStrength = np.zeros((label.shape[0], label.shape[1], label.shape[2], 2))
    labelStrength[:, :, :, 0] = label
    labelStrength[:, :, :, 1] = strength


    #lung3 = np.zeros((IProcessData.shape[0], IProcessData.shape[1], IProcessData.shape[2], 3))
    lung3 = IReadData[samples[i]+"Process"]

    # for x in range(0, label.__len__()):
    #     lung3[x, :, :, 0] = cv2.GaussianBlur(IProcessData[x, :, :].astype(np.float64), (15, 15), 0)
    #     lung3[x, :, :, 1] = cv2.equalizeHist(IProcessData[x, :, :].astype(np.uint8))
    #     lung3[x, :, :, 2] = cv2.GaussianBlur(IProcessData[x, :, :].astype(np.float64), (15, 15), 0)

    #noise = re.findall(r'\d+', i)[0]
    noise = re.findall(r'\d+', samples[i])[0]
    # plt.imshow(label[:,:,100])

    print  'segmentation ' + samples[i] + '  ->  ' + str(100*i/len(samples)) +'%'


    print str(i)+ ' Window 3'
    start = timeit.default_timer()
    segmentation = _growcut3D_cy.growcut3D(np.copy(lung3.astype("float64")), np.copy(labelStrength.astype("float64")), max_iter=10,
                                                             window_size=3)
    stop = timeit.default_timer()
    timewin3.append(stop-start)
  #  segmentation2=postProcessing(segmentation)
    #
    #
  #  GC3Dresult[Ori[0] + 'GC3D' + noise+'win3'] = segmentation2.astype('uint8')
    #
    print str(i) + 'Window 5'
    start = timeit.default_timer()
    segmentation = _growcut3D_cy.growcut3D(np.copy(lung3.astype("float64")),  np.copy(labelStrength.astype("float64")), max_iter=10,
                                                             window_size=5)
    stop = timeit.default_timer()
    timewin5.append(stop-start)

  #  segmentation2 = postProcessing(segmentation)

  #  GC3Dresult[Ori[0] + 'GC3D' + noise+'win5'] = segmentation2.astype('uint8')
    #
    print str(i) + ' Window 7'
    start = timeit.default_timer()

    segmentation = _growcut3D_cy.growcut3D(np.copy(lung3.astype("float64")),  np.copy(labelStrength.astype("float64")), max_iter=10,
                                                             window_size=7)
    stop = timeit.default_timer()
    timewin7.append(stop-start)
  #  segmentation2 = postProcessing(segmentation)
  #  GC3Dresult[Ori[0] + 'GC3D' + noise+'win7'] = segmentation2.astype('uint8')

    #
    print str(i)  + ' Window 11'
    start = timeit.default_timer()

    segmentation = _growcut3D_cy.growcut3D(np.copy(lung3.astype("float64")),  np.copy(labelStrength.astype("float64")), max_iter=10,
                                                             window_size=11)
    stop = timeit.default_timer()
    timewin11.append(stop-start)
 #   segmentation2 = postProcessing(segmentation)
  #  GC3Dresult[Ori[0] + 'GC3D' + noise+'win11'] = segmentation2.astype('uint8')

    print '-'*50








#dice = computeDiceSimilarity3D(segmentation, ground_truth.astype("uint8"))
#print ("Dice ", dice)
print("terminou")

#sio.savemat(pathOutput,GC3Dresult)



#ct_metrics.computeDiceSimilarity3D()
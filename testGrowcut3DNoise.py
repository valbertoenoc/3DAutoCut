import scipy.io as sio
import numpy as np


from growcut3D import _growcut3D_cy
import cv2
import re

from ctprocessing import ct_metrics
from matplotlib import pyplot as plt


def computeDiceSimilarity3D(array, arrayRef):
    # images must be binary
    # arrayRef = np.bitwise_not(arrayRef.astype("uint8"))
    array = array.astype("uint8")
    arrayRef[arrayRef > 1] = 255
    array[array > 1] = 255

    fitList = []
    # if len(array.shape) > 2:
    #     for i in range(0, array.shape[0]):
    #         imand = np.bitwise_and(array[i, :, :], arrayRef[i, :, :])
    #         imor = np.bitwise_or(array[i, :, :], arrayRef[i, :, :])
    #         sumand = 2 * np.sum(imand)
    #         sumor = np.sum(array[i, :, :]) + np.sum(arrayRef[i, :, :])
    #
    #         fitList.append(sumand / float(sumor))
    #
    # else:
    imand = np.bitwise_and(array.astype("uint8"), arrayRef.astype("uint8"))
    imor = np.bitwise_or(array.astype("uint8"), arrayRef.astype("uint8"))
    sumand = np.sum(imand)
    sumor = np.sum(imor)

    fitList.append(sumand / float(sumor))

    return fitList


ground_truth = sio.loadmat("data/IgroundTruth.mat")
background = sio.loadmat("data/Ibackground.mat")
ground_truth = ground_truth['Igt']
background = background['Iesfera']
pathInput="data/Ipiramide.mat"
pathOutput="data/IpiramideGC3D_growing.mat"
IReadData = sio.loadmat(pathInput)

samples = [i for i in IReadData.keys() if i.startswith('I') and i[-1].isdigit()]
Ori = [i for i in IReadData.keys() if i.startswith('I') and i[-1].isalpha()]
GC3Dresult={}

nIteration=10

for i in samples:

    IOriData = IReadData[Ori[0]]
    IProcessData = IReadData[i]

    index = 1
    spread = 0.1
    # foreground_indices  np.nonzero(ground_truth.flatten())[0]
    # background_indices = np.nonzero(~(ground_truth.flatten()))[0]

    label = np.zeros(ground_truth.shape, dtype=np.int)

    # random_foreground_indices = foreground_indices[np.random.randint(0, len(foreground_indices), spread * len(foreground_indices))]
    # random_background_indices = background_indices[np.random.randint(0, len(background_indices), spread * len(background_indices))]

    label[ground_truth > 0] = 1
    label[background > 0] = -1

    strength = np.zeros_like(IProcessData, dtype=np.float64)
    strength[label == 1] = 1.0
    strength[label == -1] = 1.0
    strength[label == 0] = 0.3

    labelStrength = np.zeros((label.shape[0], label.shape[1], label.shape[2], 2))
    labelStrength[:, :, :, 0] = label
    labelStrength[:, :, :, 1] = strength

    lung3 = np.zeros((IProcessData.shape[0], IProcessData.shape[1], IProcessData.shape[2], 3))

    for x in range(0, label.__len__()):
        lung3[x, :, :, 0] = cv2.GaussianBlur(IProcessData[x, :, :].astype(np.uint8), (3, 3), 0)
        lung3[x, :, :, 1] = cv2.GaussianBlur(IProcessData[x, :, :].astype(np.uint8), (3, 3), 0)
        lung3[x, :, :, 2] = cv2.GaussianBlur(IProcessData[x, :, :].astype(np.uint8), (3, 3), 0)

    noise = re.findall(r'\d+', i)[0]
    # plt.imshow(label[:,:,100])

    print  'segmentation ' + i


    # print i+ ' Window 3'
    # segmentation = _growcut3D_cy.growcut3D(lung3.astype("float64"), labelStrength, max_iter=nIteration,
    #                                                          window_size=3)
    # GC3Dresult[Ori[0] + 'GC3D' + noise+'win3'] = segmentation.astype('uint8')
    #
    # print i + 'Window 5'
    # segmentation = _growcut3D_cy.growcut3D(lung3.astype("float64"), labelStrength, max_iter=nIteration,
    #                                                          window_size=5)
    # GC3Dresult[Ori[0] + 'GC3D' + noise+'win5'] = segmentation.astype('uint8')
    #
    # print i + ' Window 7'
    # segmentation = _growcut3D_cy.growcut3D(lung3.astype("float64"), labelStrength, max_iter=nIteration,
    #                                                          window_size=7)
    # GC3Dresult[Ori[0] + 'GC3D' + noise+'win7'] = segmentation.astype('uint8')
    #
    # print i + ' Window 11'
    # segmentation = _growcut3D_cy.growcut3D(lung3.astype("float64"), labelStrength, max_iter=nIteration,
    #                                                          window_size=11)
    # GC3Dresult[Ori[0] + 'GC3D' + noise+'win11'] = segmentation.astype('uint8')



#dice = computeDiceSimilarity3D(segmentation, ground_truth.astype("uint8"))
#print ("Dice ", dice)
print("terminou")

sio.savemat(pathOutput,GC3Dresult)



#ct_metrics.computeDiceSimilarity3D()
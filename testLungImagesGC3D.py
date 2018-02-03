from ctprocessing import ct_lung
from ctprocessing import ct_io
import numpy as np
import scipy.io as sio
import cv2
#core processing modules
import numpy as np
import cv2
from skimage import measure
from skimage.measure import label, regionprops
from scipy import ndimage
import math
from growcut3D import _growcut3D_cy


def convert2_8bits(image16b):

    image = np.array(image16b, copy=True, dtype=np.float32)

    if image.max() > 255:
        LEVEL = -600
        WIDTH = 1500
        max = LEVEL + float(WIDTH / 2)
        min = LEVEL - float(WIDTH / 2)
        image.clip(min, max, out=image)
        image -= min
        image /= WIDTH / 255.
        image = image.astype(np.uint8)
    return image


def convertToLBP(image):
    from skimage.feature import local_binary_pattern

    radius = 2
    no_points = 8 * radius
    lbp = local_binary_pattern(image, no_points, radius, method='uniform')
    return lbp

def segmentLung3D(imageArray):
    # auxArray = np.array(imageArray, copy=True)
    auxArray = np.zeros((imageArray.shape)).astype(np.uint8)

    auxArray[imageArray < -600] = 255
    auxArray[imageArray >= -600] = 0

    limiar = 30000

    all_labels = measure.label(auxArray)
    blobs_labels = measure.label(auxArray, background=0)
    regions = measure.regionprops(blobs_labels)
    to_keep = [r for r in regions if r.area > 100000]
    lung = np.zeros((auxArray.shape))
    notLung = np.zeros((auxArray.shape))
    markers = np.zeros((auxArray.shape)).astype(np.uint8)

    lung[blobs_labels == to_keep[1].label] = 255
    notLung[blobs_labels == to_keep[0].label] = 255
    sureLung = ndimage.binary_erosion(lung).astype(np.uint8)
    probablyLung = ndimage.binary_dilation(lung).astype(np.uint8)
    probablyLung = probablyLung - sureLung

    markers[sureLung == 1] = 1
    markers[notLung == 1] = 0
    markers[probablyLung == 1] = 3

    return sureLung
    pass

def generateSeeds(lung):
    lung2 = segmentLung3D(lung)

    left = np.copy(lung2[:, :, 0:lung.shape[2] / 2])
    right = np.copy(lung2[:, :, lung.shape[2] / 2 + 1:])

    R = 15;
    seedsL = np.zeros((lung2.shape))
    seedsR = np.zeros((lung2.shape))
    # Left
    seeds=np.zeros((lung2.shape))


    [czL, cyL, cxL] = ndimage.measurements.center_of_mass(left)
    [czR, cyR, cxR] = ndimage.measurements.center_of_mass(right)
    cxR = cxR + 256

    for k in range(0, lung2.shape[0]):
        for j in range(0, lung2.shape[1]):
            for i in range(0, lung2.shape[2]):
                seedsR[k, j, i] = math.sqrt((k - czR) * (k - czR) + (j - cyR) * (j - cyR) + (i - cxR) * (i - cxR)) <= R;

    for k in range(0, lung2.shape[0]):
        for j in range(0, lung2.shape[1]):
            for i in range(0, lung2.shape[2]):
                seedsL[k, j, i] = math.sqrt((k - czL) * (k - czL) + (j - cyL) * (j - cyL) + (i - cxL) * (i - cxL)) <= R;

    seeds[seedsL > 0] = 1
    seeds[seedsR > 0] = 2

    return seeds


#Read the Dicom



#Generate the Seeds
seeds=generateSeeds(np.copy(lung))

#Define Labels
label = -1*np.ones(seeds.shape, dtype=np.float64)
label[seeds >1] = -1
label[seeds == 2] = 2

#Define Strength
strength = np.zeros(seeds.shape, dtype=np.float64)
strength[label > 0] = 1.0
strength[label < 0] = 0.1

labelStrength = np.zeros((label.shape[0], label.shape[1], label.shape[2], 2))
labelStrength[:, :, :, 0] = label
labelStrength[:, :, :, 1] = strength

processData = np.zeros((lung.shape[0], lung.shape[1], lung.shape[2], 3))

for x in range(0, label.__len__()):
    # processData[x, :, :, 0] = cv2.GaussianBlur(convert2_8bits(lung[x, :, :]).astype(np.float64), (15, 15), 0).astype(np.uint8)
    # processData[x, :, :, 1] = cv2.equalizeHist(convert2_8bits(lung[x, :, :]).astype(np.uint8)).astype(np.uint8)
    # processData[x, :, :, 2] = cv2.GaussianBlur(convert2_8bits(lung[x, :, :]).astype(np.float64), (15, 15), 0).astype(np.uint8)


    processData[x, :, :, 0] = cv2.GaussianBlur(convert2_8bits(lung[x, :, :]).astype(np.float64), (15, 15), 0).astype(np.float64)
    processData[x, :, :, 1] = cv2.equalizeHist(convert2_8bits(lung[x, :, :]).astype(np.uint8)).astype(np.float64)
    processData[x, :, :, 2] = convertToLBP(convert2_8bits(lung[x, :, :])).astype(np.float64)

print ' Window 3'
segmentation3 = _growcut3D_cy.growcut3D(np.copy(processData.astype("float64")), np.copy(labelStrength.astype("float64")),
                                       max_iter=15,
                                       window_size=15)
print 'Terminou'



ct_io.dispImages(segmentation3)

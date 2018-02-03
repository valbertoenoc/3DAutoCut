
import numpy as np
import cv2
from skimage import measure
from skimage.measure import label, regionprops
from scipy import ndimage


def computeFitAdjust3D(array, arrayRef):
    imand = np.bitwise_and(array.astype("uint8"), arrayRef.astype("uint8"))
    imor = np.bitwise_or(array.astype("uint8"), arrayRef.astype("uint8"))
    sumand = np.sum(imand)
    sumor = np.sum(imor)

    result = (sumand / float(sumor))

    return result

def computeSizeAdjust3D(array, arrayRef):
    imArea1 = np.count_nonzero(arrayRef)
    imArea2 = np.count_nonzero(array)
    subArea = np.abs(imArea1 - imArea2)
    sumArea = imArea1 + imArea2

    result = (1 - subArea / sumArea)

    return result

def computeIntensityAdjust3D(arrayOri, arrayRef, arraySeg):
    imOri = np.copy(arrayOri);
    imOri[arrayRef == 0] = 0
    inds = np.where(imOri > 0)
    meanOri = imOri[inds].mean()

    imSeg = np.copy(arrayOri);
    imSeg[arraySeg == 0] = 0
    inds = np.where(imSeg > 0)
    meanSeg = imSeg[inds].mean()

    subMean = np.abs(meanOri - meanSeg)
    sumMean = meanOri + meanSeg

    result = (1 - subMean / sumMean)

    return result

def computePositionAdjust3D(arraySeg, arrayRef):
    indsSeg = np.where(arraySeg > 0)
    indsRef = np.where(arrayRef > 0)

    centroidRefZ = indsRef[0].mean()
    centroidRefY = indsRef[1].mean()
    centroidRefX = indsRef[2].mean()

    centroidSegZ = indsSeg[0].mean()
    centroidSegY = indsSeg[1].mean()
    centroidSegX = indsSeg[2].mean()

    subCentroidZ = np.abs(centroidSegZ - centroidRefZ) / arrayRef.shape[0]
    subCentroidY = np.abs(centroidSegY - centroidRefY) / arrayRef.shape[1]
    subCentroidX = np.abs(centroidSegX - centroidRefX) / arrayRef.shape[2]

    result = 1 - (subCentroidZ + subCentroidY + subCentroidX) / 3

    return result


def computeDiceSimilarity3D(array, arrayRef):
    imand = np.bitwise_and(array.astype("uint8"), arrayRef.astype("uint8"))
    imor = np.bitwise_or(array.astype("uint8"), arrayRef.astype("uint8"))
    sumand = 2 * np.sum(imand)
    sumor = np.sum(array) + np.sum(arrayRef)

    result = (sumand / float(sumor))

    return result


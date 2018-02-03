#system access modules
import sys, os

#read CT series or .dcm/mhd/raw files
import SimpleITK as sitk
import dicom

#core processing modules
import numpy as np
import cv2
from skimage import measure
from skimage.measure import label, regionprops
from scipy import ndimage

from ctprocessing.ct_io import convert2_8bits


class Lung:
    def __init__(self):
        pass

    def segmentLung2D(self, imageArray):
            auxArray = np.zeros((imageArray.shape)).astype(np.uint8)
            imageArray = cv2.GaussianBlur(imageArray, (5, 5), 0)
            imgCanny = cv2.Canny(convert2_8bits(imageArray), 150, 200)

            auxArray[imageArray < -600] = 255
            auxArray[imageArray >= -600] = 0
            auxArray[imgCanny > 100] = 255

            all_labels = measure.label(auxArray)
            blobs_labels = measure.label(auxArray, background=0)
            regions = measure.regionprops(blobs_labels)
            regions.sort(key=lambda x: x.area, reverse=True)
            imageCenter = 256
            to_keep = [r for r in regions if ((r.area > 10)) and
                       ((r.bbox[3] - r.bbox[1]) < 512 * 0.95)]

            # to_keep = [r for r in regions if ((r.area > 20))
            # and ((r.centroid[0]>0.20*512)) and ((r.centroid[1]>0.20*512))
            # and ((r.centroid[0] < 0.85 * 512)) and ((r.centroid[1] < 0.85 * 512)) and
            #            ((r.bbox[3] - r.bbox[1]) < 512*0.95 )]

            # from pdb import set_trace
            # set_trace()
            lung = np.zeros((auxArray.shape))
            for num in range(0, to_keep.__len__()):
                lung[blobs_labels == to_keep[num].label] = 255

            airWay = self.findAirway(imageArray)
            lung[airWay == 255] = 0

            lung = ndimage.binary_closing(lung, structure=np.ones((3, 3))).astype(np.int)
            sureLung = ndimage.binary_erosion(lung).astype(np.uint8)
            probablyLung = ndimage.binary_dilation(lung, structure=np.ones((5, 5))).astype(np.int)
            probablyLung = probablyLung - sureLung
            markers = np.zeros((auxArray.shape)).astype(np.uint8)
            markers[lung >= 1] = 1

            return sureLung, markers;

    def segmentLung3D(self, imageArray):

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
        notLung= np.zeros((auxArray.shape))
        markers=np.zeros((auxArray.shape)).astype(np.uint8)

        lung[blobs_labels == to_keep[1].label] = 255
        notLung[blobs_labels == to_keep[0].label] = 255
        sureLung = ndimage.binary_erosion(lung).astype(np.uint8)
        probablyLung = ndimage.binary_dilation(lung).astype(np.uint8)
        probablyLung=probablyLung-sureLung

        markers[sureLung==1]=1
        markers[notLung == 1] = 0
        markers[probablyLung == 1] = 3

        return notLung, sureLung, markers;
        pass

    def segmentLungGrabCut(self, imageArray, markers):

            gray8b = convert2_8bits(imageArray)
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            gray = np.zeros((512, 512, 3), dtype=np.uint8)
            maskOutput = np.zeros(imageArray.shape, dtype=np.uint8)

            if len(imageArray.shape) > 2:
                for x in range(0, imageArray.__len__()):
                    print(100 * x / imageArray.__len__())
                    gray[:, :, 0] = cv2.blur(gray8b[x, :, :].astype(np.uint8),(5,5))
                    gray[:, :, 2] = cv2.equalizeHist(gray8b[x, :, :].astype(np.uint8))

                    mask = markers[x, :, :].astype(np.uint8)
                    np_counter = np.count_nonzero(mask)
                    if (np_counter > 100):
                        cv2.grabCut(gray, mask, None, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
                        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                        gray = gray * mask2[:, :, np.newaxis]
                        maskOutput[x, :, :] = mask2
            else:
                gray[:, :, 0] = cv2.blur(gray8b.astype(np.uint8), (11, 11))

                gray[:, :, 2] = cv2.equalizeHist(gray8b.astype(np.uint8))
                gray[:, :, 1] = 30*(gray8b.astype(np.uint8) - gray[:, :, 0]) + gray8b.astype(np.uint8)
                gray[:, :, 2] = self.convertToLBP(gray8b.astype(np.uint8))

                mask = markers.astype(np.uint8)
                np_counter = np.count_nonzero(mask)

                if (np_counter > 100):
                    cv2.grabCut(gray, mask, None, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
                    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                    gray = gray * mask2[:, :, np.newaxis]
                    maskOutput = mask2

            return maskOutput

    def convertToLBP(self, image):
        from skimage.feature import local_binary_pattern

        radius = 2
        no_points = 8 * radius
        lbp = local_binary_pattern(image, no_points, radius, method='uniform')
        return lbp

    def findAirway(self, originalImage):

        win=70
        imageCenter = 256

        auxArray = np.zeros((originalImage.shape)).astype(np.uint8)
        auxArray[originalImage < -600] = 255
        auxArray[originalImage >= -600] = 0
        auxArray = auxArray[(imageCenter-win):(imageCenter+win),(imageCenter-win):(imageCenter+win)]

        blobs_labels = measure.label(auxArray, background=0)

        regions = measure.regionprops(blobs_labels)

        regions.sort(key=lambda x:x.area,reverse=True)
        to_keep = [r for r in regions if ((r.eccentricity < 0.75)) and ((r.eccentricity > 0.0)) and (r.area<1200) and (r.area>300) and (r.bbox[3]<2*win*0.9) and (r.bbox[2]<2*win*0.9)
        and (r.bbox[0] > 2 * win * 0.1) and (r.bbox[1] > 2 * win * 0.1)]

        airWay = np.zeros((auxArray.shape))

        for num in range(0, to_keep.__len__()):
            airWay[blobs_labels == to_keep[num].label] = 255

        auxArray = np.zeros((originalImage.shape)).astype(np.uint8)
        auxArray[(imageCenter-win):(imageCenter+win),(imageCenter-win):(imageCenter+win)]=np.copy(airWay)

        return auxArray

    def computeFitArea(self, array, arrayRef):
        #images must be binary
        arrayRef = np.bitwise_not(arrayRef.astype("uint8"))
        arrayRef[arrayRef >= 1] = 255

        fitList = []
        if len(array.shape) > 2:
            for i in range(0, array.shape[0]):
                imand = np.bitwise_and(array[i, :, :].astype("uint8"), arrayRef[i, :, :].astype("uint8"))
                imor = np.bitwise_or(array[i, :, :].astype("uint8"), arrayRef[i, :, :].astype("uint8"))
                sumand = np.sum(imand)
                sumor = np.sum(imor)

                fitList.append(sumand / float(sumor))

        else:
            imand = np.bitwise_and(array.astype("uint8"), arrayRef.astype("uint8"))
            imor = np.bitwise_or(array.astype("uint8"), arrayRef.astype("uint8"))
            sumand = np.sum(imand)
            sumor = np.sum(imor)

            fitList.append(sumand / float(sumor))

        return fitList

    def computeDiceSimilarity(self, array, arrayRef):
        #images must be binary
        arrayRef = np.bitwise_not(arrayRef.astype("uint8"))
        array = array.astype("uint8")
        arrayRef[arrayRef > 1] = 255

        fitList = []
        if len(array.shape) > 2:
            for i in range(0, array.shape[0]):
                imand = np.bitwise_and(array[i, :, :], arrayRef[i, :, :])
                imor = np.bitwise_or(array[i, :, :], arrayRef[i, :, :])
                sumand = 2*np.sum(imand)
                sumor = np.sum(array[i, :, :]) +np.sum(arrayRef[i, :, :])

                fitList.append(sumand / float(sumor))

        else:
            imand = np.bitwise_and(array.astype("uint8"), arrayRef.astype("uint8"))
            imor = np.bitwise_or(array.astype("uint8"), arrayRef.astype("uint8"))
            sumand = np.sum(imand)
            sumor = np.sum(imor)

            fitList.append(sumand / float(sumor))

        return fitList

    def postProcessing(self, exam):

        auxArray = np.copy(exam)

        blobs_labels = measure.label(auxArray, background=0)
        regions = measure.regionprops(blobs_labels)
        regions.sort(key=lambda x:x.area,reverse=True)

        to_keep = [r for r in regions if ((r.area > 1000)) ]

        lung = np.zeros((auxArray.shape))
        for num in range(0, to_keep.__len__()):
            lung[blobs_labels == to_keep[num].label] = 255

        lung = ndimage.binary_fill_holes(lung).astype(np.int)
        lung = ndimage.binary_opening(lung).astype(np.int)
        lung = ndimage.binary_closing(lung, structure=np.ones((3, 3))).astype(np.int)

        return lung


import scipy.io as sio
import numpy as np
import timeit

# from growcut3D import _growcut3D_cy
import morphsnakes

import cv2
import re

#from ctprocessing import ct_metrics
from matplotlib import pyplot as plt

ground_truth = sio.loadmat("data/IgroundTruth.mat")
background = sio.loadmat("data/Ibackground.mat")
ground_truth = ground_truth['Igt']
background = background['Iesfera']

data=['Icubo']
time=[]
for k in range(0, data.__len__()):

    pathInput="data/"+data[k]+".mat"
    pathOutput="data/"+data[k]+"MACWE3D_noise.mat"
    pathOutput2="data/"+data[k]+"MGAC3D_noise.mat"


    IReadData = sio.loadmat(pathInput)

    samples = [i for i in IReadData.keys() if i.startswith('I') and i[-1].isdigit()]
    Ori = [i for i in IReadData.keys() if i.startswith('I') and i[-1].isalpha()]
    MACWEresult={}
    MGACresult={}
    nIteration=200

    for i in xrange(0,len(samples)):

        IOriData = IReadData[samples[i]]

        noise = re.findall(r'\d+', samples[i])[0]
        #noise = "0"
        # plt.imshow(label[:,:,100])

        print  'segmentation ' + str(i)

        start = timeit.default_timer()
        macwe = morphsnakes.MorphACWE(IOriData)
        macwe.levelset = np.copy(ground_truth)
        macwe.run(10)
        stop = timeit.default_timer()
        print stop - start
        time.append(stop - start)
    ##    MACWEresult[Ori[0] + 'MACWE3D' + noise+'win0'] = np.copy(macwe.levelset)


        # start = timeit.default_timer()
        # mgac = morphsnakes.MorphGAC(IOriData)
        # mgac.levelset = np.copy(ground_truth)
        #
        # gI = morphsnakes.gborders(IOriData, alpha=1000, sigma=2)
        # mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.31, balloon=1)
        #
        # # mgac = morphsnakes.MorphGAC(IOriData)
        # mgac.levelset = np.copy(ground_truth)
        #
        # mgac.run(10)
        # stop = timeit.default_timer()
        #
        # print stop - start
        # time.append(stop - start)
      #  MGACresult[Ori[0] + 'MGAC3D' + noise + 'win0'] = np.copy(mgac.levelset)

        print '-'*50


    #dice = computeDiceSimilarity3D(segmentation, ground_truth.astype("uint8"))
    #print ("Dice ", dice)
    print("terminou")

  ##  sio.savemat(pathOutput,MACWEresult)

  #  sio.savemat(pathOutput2,MGACresult)


#ct_metrics.computeDiceSimilarity3D()
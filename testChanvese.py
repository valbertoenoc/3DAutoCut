import scipy.io as sio
import numpy as np
import timeit

# from growcut3D import _growcut3D_cy
import chanvese3d

import cv2
import re

#from ctprocessing import ct_metrics
from matplotlib import pyplot as plt

ground_truth = sio.loadmat("data/IgroundTruth.mat")
background = sio.loadmat("data/Ibackground.mat")
ground_truth = ground_truth['Igt']
background = background['Iesfera']

data=['Icilindro','Iduplocone','Iesfera','Icubo']

for k in range(0, data.__len__()):

    pathInput="data/"+data[k]+".mat"
    pathOutput="data/"+data[k]+"Chanvese3D_noise.mat"


    IReadData = sio.loadmat(pathInput)

    samples = [i for i in IReadData.keys() if i.startswith('I') and i[-1].isdigit()]
    Ori = [i for i in IReadData.keys() if i.startswith('I') and i[-1].isalpha()]
    ChanveseResult={}
    nIteration=200

    for i in xrange(0,len(samples)):

        IOriData = IReadData[samples[i]]

        noise = re.findall(r'\d+', samples[i])[0]
        #noise = "0"
        # plt.imshow(label[:,:,100])

        print  'segmentation ' + str(i)


    ##    macwe = morphsnakes.MorphACWE(IOriData)
    ##    macwe.levelset = np.copy(ground_truth)
    ##    macwe.run(nIteration)
    ##    MACWEresult[Ori[0] + 'MACWE3D' + noise+'win0'] = np.copy(macwe.levelset)
        start = timeit.default_timer()

        seg, phi, its= chanvese3d(IOriData, ground_truth, max_its=200, alpha=0.2, thresh=0, color='r', display=False)

        stop = timeit.default_timer()

        print stop - start
        ChanveseResult[Ori[0] + 'Chanvese3D' + noise + 'win0'] = np.copy(seg)

        print '-'*50


    #dice = computeDiceSimilarity3D(segmentation, ground_truth.astype("uint8"))
    #print ("Dice ", dice)
    print("terminou")

    sio.savemat(pathOutput,ChanveseResult)

  #  sio.savemat(pathOutput2,MGACresult)


#ct_metrics.computeDiceSimilarity3D()
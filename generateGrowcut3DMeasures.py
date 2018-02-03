import scipy.io as sio
import xlwt
import numpy as np
from growcut3D import _growcut3D_cy
import cv2
import re

from ctprocessing import ct_metrics

data=['Iduplocone','Icubo','Iesfera','Icilindro']
#data=['Icilindro']
book = xlwt.Workbook(encoding="utf-8")
pathSave="output/Results_MGAC3D.xls"
for k in range(0, data.__len__()):

    pathInput="data/"+data[k]+".mat"
    pathOutput="data/"+data[k]+"MGAC3D_noise.mat"
    IReadInputData = sio.loadmat(pathInput)
    IReadOutputData = sio.loadmat(pathOutput)

    gt = [i for i in IReadInputData.keys() if i.startswith('I') and i[-1].isalpha() and not any(j.isdigit() for j in i)]
    samples = [i for i in IReadOutputData.keys() if i.startswith('I') and i[-1].isdigit() ]
    Igt = IReadInputData[gt[0]]
    type = [i for i in IReadInputData.keys() if i.startswith('I') and i[-1].isalpha() and not any(j.isdigit() for j in i)]



    print "Tipo","Janela","Ruído"


    sheet1 = book.add_sheet(type[0])
    sheet1.write(0,0,"Tipo")
    sheet1.write(0,1,"Janela")
    sheet1.write(0,2,"Ruído")
    sheet1.write(0,3,"Dice")
    sheet1.write(0,4,"Fit")
    sheet1.write(0,5,"Position")
    sheet1.write(0,6,"Size")



    Igt[Igt > 0] = 1

    for i in range(0,samples.__len__()):
    #for i in range(0,3):
        print i
        OutputData = IReadOutputData[samples[i]]
        name = samples[i]
        type = samples[i][0:name.find('3D') - 3]
        window = samples[i][name.find('win') + 3:]
        window=0
        noise = name[name.find('3D') + 2:len(name)]
        #Calculate the metrics

        OutputData[OutputData>250]=1

        Dice = ct_metrics.computeDiceSimilarity3D(OutputData.astype("uint8"), Igt.astype("uint8"))
        sheet1.write(i+1, 3, Dice)
        Fit=ct_metrics.computeFitAdjust3D(OutputData.astype("uint8"),Igt[:,:].astype("uint8"))
        sheet1.write(i + 1, 4, Fit)
        Position=ct_metrics.computePositionAdjust3D(OutputData.astype("uint8"),Igt.astype("uint8"))
        sheet1.write(i + 1, 5, Position)
        Size=ct_metrics.computeSizeAdjust3D(OutputData.astype("uint8"),Igt.astype("uint8"))
        sheet1.write(i + 1, 6, Size)

        sheet1.write(i + 1,0,type)
        sheet1.write(i + 1, 1, window)
        sheet1.write(i + 1, 2, noise)

book.save(pathSave)
print "fim"

import cv2
import os
from lxml import etree

from object_detection.utils import dataset_util

metaDir = "H:\\DataSet\\Main\\학습데이터버전관리\\2019-04-10"

#file 
metaLst = os.listdir(metaDir)

for metaName in metaLst:
    metaPath = os.path.join(metaDir, metaName)
    with open(metaPath,"r") as metaFile:
        metaData = metaFile.read()
        print(metaData)





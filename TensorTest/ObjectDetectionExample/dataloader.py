import xml.etree.ElementTree as ET
from abc import abstractmethod, ABCMeta
import numpy as np
import cv2
import os
import glob
import json

IMG_EXTENSIONS = ['png','jpg','bmp']

class DataSetBase(metaclass=ABCMeta):

    def __init__(self,imgShape):
        self.imgShape = imgShape

    def export_class_info(self,clsInfo):

        with open('anno.json','w') as outData:
            json.dumps(clsInfo,outData)

    @abstractmethod
    def load_class_info(self,clsInfoDir):
        pass
    #id : int32
    #displayName : string

    # testDic = { 'class' : [{'id':5,'name':'dog'},{'id':4,'name':'cat'},{'id':3,'name':'Lion'}]}
    # jsonStr = json.dumps(testDic)
    # jsonArry = json.loads(jsonStr)

    # for item in jsonArry['class']:
    #     print(item['id'])
    #     print(item['name'])

        # with open(clsInfoDir,'r') as data:
        #   rslt = json.load(data)

        # return rslt['category']
    @abstractmethod
    def load_dataset(self,metaPath, imgPath,clsInfoDir):
        pass

class DataSetPascalVOC(DataSetBase):

    def __init__(self,imgShape):
        super(DataSetPascalVOC,self).__init__(imgShape)

    def load_class_info(self,clsInfoDir):
        
        with open(clsInfoDir,'r') as data:
          rslt = json.load(data)

        return rslt['category']


    def load_dataset(self,metaPath,imgPath,clsInfoDir):
        imgSet = []
        metaSet = []
        imgPaths =[]

        clsInfo = self.load_class_info(clsInfoDir)

        for ext in IMG_EXTENSIONS:
            imgPaths.append(glob.glob(os.path.join(imgPath,"*.{}".format(ext))))

        for imgPath in imgPaths:
            im = cv2.imread(imgPath)
            im = np.array(im,dtype=np.float32) #type casting
            im = cv2.resize(im,(self.imgShape[1],self.imgShape[0]))
            imgSet.append(im)

            metaName = os.path.splitext(os.path.basename(imgPath))[0]
            annoPath = os.path.join(metaPath,'{}.xml'.format(metaName))

            anno = ET.parse(annoPath).getroot()
            for i in anno.findall('object'):

                for cId, cName in clsInfo.items():
                    if cName not in i.get('name'):
                        continue
                
                x_min, y_min,x_max,y_max=i.get('x_min','y_min','x_max','y_max')
                #cls id 필요함
                metaSet.append(np.array(np.float32(cId),np.float32(x_min),np.float32(y_min)
                             ,np.float32(x_max),np.float32(y_max)))
                #print(i.find('name').text)

        return imgSet, metaSet#image와 meta가 잘 매칭되어야한다. 

class DataSetAnnoExtension(DataSetBase):

    def __init__(self,imgShape):
        super(DataSetAnnoExtension,self).__init__(imgShape)

    def load_class_info(self,clsInfoDir):
        with open(clsInfoDir,'r') as data:
          rslt = json.load(data)

        return rslt['category']

    def load_json(self,path):
        with open(path,'r') as data:
            rslt = json.load(data)

        return rslt

    def load_dataset(self,metaPath, imgPath,clsInfoDir):
        imgSet = []
        metaSet = []
        imgPaths =[]

        # clsInfo = self.load_class_info(clsInfoDir) 
        # 이거 왜 리스트로 나오는지 원인 파악 필요함.
        with open(clsInfoDir,'r') as data:
          clsInfo = json.load(data)

        for ext in IMG_EXTENSIONS:
            path=os.path.join(imgPath,"*.{}".format(ext))
            fullPath = glob.glob(path)

            if not fullPath:
                continue

            imgPaths.extend(fullPath)

        for imgPath in imgPaths:
            im = cv2.imread(imgPath)
            im = np.array(im,dtype=np.float32) #type casting
            im = cv2.resize(im,(self.imgShape[1],self.imgShape[0]))
            imgSet.append(im)

            metaName = os.path.splitext(os.path.basename(imgPath))[0]
            annoPath = os.path.join(metaPath,'{}.anno'.format(metaName))
            anno = self.load_json(annoPath)

            for item in clsInfo['category']:
                idx = item['id']
                name = item['name']

                if name not in anno:
                    continue
                
                for x_min, y_min, x_max, y_max in anno[name]:
                    arry = np.array([np.float32(idx),np.float32(x_min),np.float32(y_min)
                             ,np.float32(x_max),np.float32(y_max)])
                    metaSet.append(arry)

        return imgSet, metaSet#image와 meta가 잘 매칭되어야한다. 


# if __name__ =='__main__':
    # load_dataset(path='',imgSize='',clsInfoDir='')
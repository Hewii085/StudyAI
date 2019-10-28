from dataloader import DataSetAnnoExtension
from layer import TestModel

def main():
    metaPath = 'F:\\ReferenceSources\\OpenSource\\MyPrivateSource\\TensorStudy\\src\\Yolo\\data\\face\\train\\annotations'
    imgPath  = 'F:\\ReferenceSources\\OpenSource\\MyPrivateSource\\TensorStudy\\src\\Yolo\\data\\face\\train\\images'
    clsInfoPath = '.\\anno.json'
    imgSize = [250,250]

    dataSet = DataSetAnnoExtension(imgSize)

    imgSet, metaSet = dataSet.load_dataset(metaPath,imgPath,clsInfoPath)
    trainer = TestModel(10,5000)
    trainer.train(imageSet=imgSet,metaSet=metaSet)
    
    print("Finished")
    #trainer tensorflow session / saver module
    
if __name__ == '__main__':
    main()
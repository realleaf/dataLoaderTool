import torch
import torch.cuda
import numpy as np
import pandas as pd
from itertools import chain

dataAddress = '/home/diamonds/Documents'
dataModel = ['train', 'test']
numFiles = 20
channelType = ['I', 'Q']

class myData():
    def collectData(self, channelType, Model='train', numFiles=20,  # tested, OK. receive tensor only
                    dataAddress='/home/diamonds/Documents'):
        deviceNameList = ['A1']
        cols = 1 if channelType == 'I' else 2
        deviceNameList = deviceNameList if Model == 'train' else deviceNameList[:4]
        labelMark = []
        labelDict = {'A1': 0,
                     'A2': 10,
                     'B1': 20,
                     'B2': 30,
                     'C1': 40,
                     'C2': 50}
        landMark = False
        for dev in deviceNameList:
            for i in range(numFiles):
                dataBuffer = pd.read_excel(
                    '%s/data/processed_data/%s/%s/%s_%d.xlsx' % (dataAddress, Model, dev, dev, i + 1), usecols=[cols])
                dataBuffer = np.array(dataBuffer)
                dataBuffer = torch.Tensor(dataBuffer)
                labelMark.append(labelDict.get(dev))
                if landMark:
                    dataVector = torch.cat((dataVector, dataBuffer), 1)
                else:
                    dataVector = dataBuffer
                landMark = True
        labelMark = [labelMark]
        # labelMark = np.array(labelMark)
        labelMark = torch.Tensor(labelMark)
        dataVector = torch.cat((dataVector, labelMark), 0)
        return dataVector

    def splitData(self, originalDataset, splitScala=10000):  # tested, OK
        numCol = originalDataset.size(1)
        numRow = originalDataset.size(0)
        segNum = numRow // splitScala
        labels = originalDataset[numRow - 1, :]
        newLabels = []
        labels = labels.numpy()
        for i in labels:
            tmp = [i] * segNum
            newLabels.append(tmp)
        newLabels = list(chain(*newLabels))
        newLabels = torch.Tensor(newLabels).reshape(1,
                                                    len(newLabels))  # ************************************************
        originalDataset = originalDataset[:segNum * splitScala].t()
        tempData = originalDataset.reshape(segNum * numCol, splitScala).t()
        tempData = torch.cat((newLabels, tempData), 0)
        return tempData

    def dataLoader(self, dataset, batchSize, Modle='test'):  # tested, OK
        shuffle = True if Modle == 'train' else False
        _b = []
        iter = dataset.size(1) // batchSize
        for i in range(iter):
            if shuffle:
                arr = dataset[:, batchSize * i:batchSize * (i + 1)]
                _d = arr.t().numpy()
                np.random.shuffle(_d)
                _b.append(torch.from_numpy(_d).t())
            else:
                _b.append(dataset[:, batchSize * i:batchSize * (i + 1)])
        if not iter == len(dataset[0]) / batchSize:
            if shuffle:
                arr = dataset[:, batchSize * iter:]
                _d = arr.t().numpy()
                np.random.shuffle(_d)
                _b.append(torch.from_numpy(_d).t())
            else:
                _b.append(dataset[:, batchSize * iter:])
        return _b  # list of tensor

 # tested, OK.
dataTool = myData()

trainDataSet = dataTool.collectData(
    channelType=channelType[0]
)
t1 = dataTool.splitData(
    originalDataset=trainDataSet,
    splitScala=10000
)
t2 = dataTool.dataLoader(
    dataset=t1,
    batchSize=10,
    Modle='train'
)
print(len(t2))
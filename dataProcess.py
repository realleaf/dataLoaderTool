import pandas as pd
import numpy as np

#parameters to modify first
deviceNameTrain = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
deviceNameTest = ['A1', 'A2', 'B1', 'B2']
numFiles = 20
dataModel = ['train', 'test']
dataAddress = '/home/diamonds/Documents'   #just end without '/'

class dataP():
    def process(self, rawData):
        mean_ = np.mean(rawData)
        stand_ = np.std(rawData, ddof=1)
        temp = (rawData - mean_) / stand_
        max_ = max(temp)
        min_ = min(temp)
        return (temp - min_) / (max_ - min_)

    def doProcessing(self, rawDataset):
        processedData = []
        for i in range(2):
            processedData.append(list(self.process(rawData=rawDataset[:, i])))
        return np.array(processedData).transpose()

datap = dataP()

for m in dataModel:
    deviceName = deviceNameTrain if m == 'train' else deviceNameTest
    for dev in deviceName:
        for i in range(numFiles):
            #support recover from breakpoint
            if not os.path.exists('%s/data/processed_data/%s/%s/%s_%d.xlsx' % (dataAddress, m, dev, dev, i + 1)):
                #receive raw data
                dataReceiver = pd.read_excel('%s/data/raw_data/%s/%s/%s_%d.xlsx'%(dataAddress, m, dev, dev, i+1), usecols=[1,3])
                rawDataset = np.array(dataReceiver)
                #process data and save as processed data
                processedData = datap.doProcessing(rawDataset=rawDataset)
                writer = pd.ExcelWriter('%s/data/processed_data/%s/%s/%s_%d.xlsx'%(dataAddress, m, dev, dev, i+1))
                df = pd.DataFrame(data={'I':processedData[:,0],"Q":processedData[:,1]})
                df.to_excel(writer)
                writer.save()

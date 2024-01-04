import json

class Config:
    def __init__(self):
        self.imageShape=(1228,1228,3)
        self.inputShape=(256,256,3)
        self.pretrainedModel=None
        self.classes=3
        self.model="PathoNet" 
        self.logPath="logs/"
        self.data_path=""
        self.loss="mse"
        self.optimizer="adam"
        self.lr=1e-2
        self.batchSize=16
        self.epoches=30
        self.validationSplit=0.2
        self.trainDataPath=""
        self.thresholds=[120,180,40]
        self.guaMaxValue=255
        self.minDistance =5
    
    def load(self,configPath):
        with open(configPath,'r') as f:
            confDict = json.load(f)
            self.__dict__.update(confDict)
    def update_thresold_1(self, thr1):
        self.thresholds[1] = thr1
    def update_thresold_0(self, thr0):
        self.thresholds[0] = thr0
    
    def save(self,configPath):
        with open(configPath, 'w') as f:
            json.dump(self.__dict__, f)

            
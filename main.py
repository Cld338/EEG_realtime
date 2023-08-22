from local_libs.featureExtraction import *
from local_libs.openvibe_tool import *
from local_libs.private_tool import *
from local_libs.dataAnalyzer import *
import numpy as np
import warnings
import joblib

class MIClassifier():
    def __init__(self) -> None:
        self.lsl = LSL()
        currDir = os.getcwd()
        self.svm_model = joblib.load(currDir+'/src/model/svm.pkl') 
        self.lda_coef = np.load(f"{currDir}/src/data/lda_coef.npy")
        self.left_csp_filter = np.load(currDir+"/src/data/left_csp_filter.npy")
        self.right_csp_filter = np.load(currDir+"/src/data/right_csp_filter.npy")
        self.tongue_csp_filter = np.load(currDir+"/src/data/tongue_csp_filter.npy")
        self.foot_csp_filter = np.load(currDir+"/src/data/foot_csp_filter.npy")

    def connect(self)->None:
        self.lsl.connect()
        self.signal_ls = []

    def classify(self, data):
        
        data_by_channel = [[] for i in range(24)]
        for i in range(len(data)):
            if data[i][0]:
                for j in range(len(data[i][0])):
                    data_by_channel[j].append(data[i][0][j])

        data_by_channel = np.array([np.array([np.array(j) for j in i]) for i in data_by_channel])

        data_by_channel = np.asarray([bandpass_filter(i, 300, 7, 30) for i in data_by_channel])
        data_left_csp = np.asarray(data_by_channel).T@self.left_csp_filter
        data_right_csp = np.asarray(data_by_channel).T@self.right_csp_filter
        data_tongue_csp = np.asarray(data_by_channel).T@self.tongue_csp_filter
        data_foot_csp = np.asarray(data_by_channel).T@self.foot_csp_filter

        feature_left = twoMDimensionalFeature([data_left_csp], 23, 1)
        feature_right = twoMDimensionalFeature([data_right_csp], 23, 1)
        feature_tongue = twoMDimensionalFeature([data_tongue_csp], 23, 1)
        feature_foot = twoMDimensionalFeature([data_foot_csp], 23, 1)
        
        LDA_left = feature_left@self.lda_coef.T
        LDA_right = feature_right@self.lda_coef.T
        LDA_tongue = feature_tongue@self.lda_coef.T
        LDA_foot = feature_foot@self.lda_coef.T
        classes = ["left", "right", "tongue", "foot"]
        probability = [round(self.svm_model.predict_proba(data)[0][idx], 8) for idx, data in enumerate([LDA_left, LDA_right, LDA_tongue, LDA_foot])]
        print({classes[i]:probability[i] for i in range(4)})
        predict_idx = probability.index(max(probability))
        print(["left", "right", "tongue", "foot"][predict_idx])

    def start(self, windowLength, overlap):
        self.lsl.inlet.flush()
        dFrame = 0
        flag = False
        while True:
            signal = self.lsl.receiveData()
            self.signal_ls.append(signal)
            dFrame += 1
            if flag:
                if dFrame >= overlap:
                    self.classify(self.signal_ls[-windowLength:])
                    dFrame = 0
            else:   
                if dFrame == windowLength:
                    flag = True

if __name__=="__main__":
    warnings.filterwarnings(action='ignore')
    clf = MIClassifier()
    clf.connect()
    clf.start(windowLength=1200,
              overlap=900)
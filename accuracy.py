from local_libs.featureExtraction import *
from local_libs.openvibe_tool import *
from local_libs.private_tool import *
from local_libs.dataAnalyzer import *
from PIL import Image, ImageTk
from tkinter import PhotoImage
import multiprocessing as mp
from sklearn.svm import SVC
import tkinter as tk
import pandas as pd
import numpy as np
import threading
import itertools
import warnings
import random
import joblib
import os
import os
import sys


def model():
    LDA_DF = pd.read_csv(currDir+"/src/data/LDA_DF.csv")
    # 데이터 생성 (여기서는 임의로 데이터를 생성하지만, 실제 데이터를 사용하시면 됩니다)
    df = pd.DataFrame(LDA_DF)

    # 데이터와 레이블 분리
    X = df.drop('label', axis=1)
    y = df['label']

    # 데이터 분할
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # # SVM 모델 생성 및 학습
    svm_model = SVC(kernel='linear', C=1.0, random_state=100, probability=True)
    svm_model.fit(X, y)
    joblib.dump(svm_model, currDir+'/src/model/svm.pkl') 
    # # 예측
    # y_pred = svm_model.predict(X_test)
    return svm_model



def classify(data):
    global svm_model
    global left_csp_filter, right_csp_filter, tongue_csp_filter, foot_csp_filter
    data_by_channel = [[] for i in range(24)]
    for i in range(len(data)):
        if data[i][0]:
            for j in range(len(data[i][0])):
                data_by_channel[j].append(data[i][0][j])

    # print(data_by_channel)
    data_by_channel = np.array([np.array([np.array(j) for j in i]) for i in data_by_channel])

    data_by_channel = np.asarray([bandpass_filter(i, 300, 7, 30) for i in data_by_channel])
    data_left_csp = np.asarray(data_by_channel).T@left_csp_filter
    data_right_csp = np.asarray(data_by_channel).T@right_csp_filter
    data_tongue_csp = np.asarray(data_by_channel).T@tongue_csp_filter
    data_foot_csp = np.asarray(data_by_channel).T@foot_csp_filter

    feature_left = twoMDimensionalFeature([data_left_csp], 23, 1)
    feature_right = twoMDimensionalFeature([data_right_csp], 23, 1)
    feature_tongue = twoMDimensionalFeature([data_tongue_csp], 23, 1)
    feature_foot = twoMDimensionalFeature([data_foot_csp], 23, 1)
    
    LDA_left = feature_left@lda_coef.T
    LDA_right = feature_right@lda_coef.T
    LDA_tongue = feature_tongue@lda_coef.T
    LDA_foot = feature_foot@lda_coef.T

    probability = [svm_model.predict_proba(data)[0][idx] for idx, data in enumerate([LDA_left, LDA_right, LDA_tongue, LDA_foot])]
    print(probability)
    predict_idx = probability.index(max(probability))
    print(["left", "right", "tongue", "foot"][predict_idx])
    print(svm_model.predict(LDA_left), "!!!!!!!")

def random_sequence():
    seq = [0, 1, 2, 3] # 상하좌우
    arr1 = list(itertools.permutations(seq, 4))*8
    random.shuffle(arr1)
    print(arr1[:180])
    np.save(f"{currDir[:-11]}/src/data/arr.npy", np.array(arr1[:180]))

class ArrowDisplayApp:
    def __init__(self, root, arr):
        # self.trackIdx = len(filesInFolder(currDir+"/src/data/ohjihun"))+1
        self.trackIdx = 0
        self.lsl = LSL()
        self.lsl.connect()
        self.signal_ls = []
        self.root = root
        self.root.title("Arrow Display App")
        self.root.attributes('-fullscreen', True)
        self.currDir = os.getcwd()
        self.arrow_images = [
            self.currDir+"/src/images/arrow_up.png",
            self.currDir+"/src/images/arrow_down.png",
            self.currDir+"/src/images/arrow_left.png",
            self.currDir+"/src/images/arrow_right.png"
        ]
        self.session = 0
        self.started = False
        self.arr = arr
        self.current_arrow_index = 0
        self.arrow_label = None
        self.a = None
        self.dot_label = None
        self.btnStart = None
        self.root.configure(bg='white')
        self.update_arrow_image()
        self.root.after(0, self.initial_window)

    def process_receive(self):
        self.lsl.inlet.flush()
        dFrame = 0
        flag = False
        # print("!!!!!!!!!!")
        while True:
            signal = self.lsl.receiveData()
            self.signal_ls.append(signal)
            dFrame += 1
            if flag:
                if dFrame >= 900:
                    classify(self.signal_ls[-1200:])
                    dFrame = 0
            else:   
                if dFrame == 1200:
                    flag = True

    def clear_window(self): 
        if self.arrow_label:
            self.arrow_label.destroy()
        if self.a:
            self.a.destroy()
        if self.dot_label:
            self.dot_label.destroy()
        if self.btnStart:
            self.btnStart.destroy()

    def btnStartCmd(self):
        # self.p1 = mp.Process(target=self.process_receive, name="receiver", args=[signal_ls])
        # self.p1.start()
        self.thread = threading.Thread(target=self.process_receive)
        self.thread.start()
        # self.started = Truex
        self.display_none(10000, 1)

    def initial_window(self):
        self.clear_window()
        self.a = tk.Label(self.root, width=200, height=26, background="white")
        self.a.pack()
        self.btnStart = tk.Button(self.root, width=12, height=4,text="Start", command=self.btnStartCmd)
        self.btnStart.pack()

    def update_arrow_image(self):
        image_path = self.arrow_images[self.arr[self.session+6*(self.trackIdx-1)][self.current_arrow_index-1]]
        arrow_image = Image.open(image_path)
        ratio = 0.5
        arrow_image = arrow_image.resize((round(arrow_image.size[0]*ratio), round(arrow_image.size[1]*ratio)), Image.ANTIALIAS)
        self.arrow_photo = ImageTk.PhotoImage(arrow_image)
        dot_image = Image.open(self.currDir+"/src/images/dot.png")
        dot_image = dot_image.resize((766, 766), Image.ANTIALIAS)
        self.dot_photo = ImageTk.PhotoImage(dot_image)

    def display_next_arrow(self):
        if self.current_arrow_index == 5:
            self.root.after(4000, self.display_none, 3000, 2)
            self.current_arrow_index = 0
            self.session += 1
            if self.session==6:
                # saveJson(f"{self.currDir}/src/data/ohjihun/Track{self.trackIdx}_chair.json", list(self.signal_ls))
                self.root.destroy()
                self.root.quit()
        else:   
            self.update_arrow_image()
            self.a = tk.Label(self.root, width=200, height=12, background="white")
            self.a.pack()
            self.arrow_label = tk.Label(self.root, image=self.arrow_photo, background="white")
            self.arrow_label.pack()
            self.current_arrow_index += 1
            self.root.after(4000, self.display_none, 3000, 2)

    def display_none(self, ms, flag):
        self.clear_window()
        if flag==1:
            """세션 종료"""
            self.root.after(ms, self.display_dot)
        else:
            """Trial 종료"""
            self.root.after(ms, self.display_next_arrow)

    def display_dot(self):
        self.clear_window()
        self.dot_label = tk.Label(self.root, image=self.dot_photo, background="white")
        self.dot_label.pack()

        self.root.after(7000, self.display_next_arrow)


if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')
    currDir = os.getcwd()
    manager = mp.Manager()
    arr = np.load(f"{currDir}/src/data/arr.npy")
    lda_coef = np.load(f"{currDir}/src/data/lda_coef.npy")
    svm_model = model()
    left_csp_filter = np.load(currDir+"/src/data/left_csp_filter.npy")
    right_csp_filter = np.load(currDir+"/src/data/right_csp_filter.npy")
    tongue_csp_filter = np.load(currDir+"/src/data/tongue_csp_filter.npy")
    foot_csp_filter = np.load(currDir+"/src/data/foot_csp_filter.npy")
    root = tk.Tk()
    app = ArrowDisplayApp(root, arr)
    root.mainloop()
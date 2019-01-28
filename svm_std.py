#coding: UTF-8

###############################################################
#
# version
#  1.0 2018/11/01
#
###############################################################

import kayaru_standard_process as kstd                              
import kayaru_standard_messages as kstdm
import kayaru_standard_process_for_image as image                   # ver 1.0
import kayaru_standard_process_for_randomize as rand                # ver 0.1
import numpy as np
import pandas as pd
import os
import joblib as jl

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

def getLabelList(dtoNT):

    data      = dtoNT.getVariable()
    dtoNL_ans = kstd.DtoNpList()

    for ri in range(dtoNT.getAttrRowLength()):
        for ci in range(dtoNT.getAttrColLength()):
            if data[ri][ci] == 1:
                dtoNL_ans.add(ci)

    return dtoNL_ans.getVariable()


class ReadData():
    def __init__(self):
        digits = load_digits()
        train_x, test_x, train_y, test_y  = train_test_split(digits.data, digits.target)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x  = test_x
        self.test_y  = test_y

    def getTrainX(self):
        return self.train_x

    def getTrainY(self):
        return self.train_y

    def getTestX(self):
        return self.test_x

    def getTestY(self):
        return self.test_y

if __name__ == "__main__":

    process = "c-svc"

    mq = kstdm.MessQueue()
    sub_process = "reading file : train_x"
    mq.setMess(sub_process)
    sub_process = "reading file : train_y"
    mq.setMess(sub_process)
    sub_process = "during estimating"
    mq.setMess(sub_process)
    sub_process = "reading file : test_x"
    mq.setMess(sub_process)

    kstd.echoStart(process)

    C = 1.0
    kernel = 'rbf'
    gamma  = 0.01

    data = ReadData()

    train_x = data.getTrainX()
    kstd.echoIsAlready(mq.getMess())

    train_y = data.getTrainY()
    kstd.echoIsAlready(mq.getMess())

    estimator  = SVC(C=C, kernel=kernel, gamma=gamma)
    classifier = OneVsRestClassifier(estimator)
    classifier.fit(train_x, train_y)

    test_x = data.getTestX()
    test_y = data.getTestY()

    predict_y  = classifier.predict(test_x)

    print(accuracy_score(test_y, predict_y))

    output_file_path = kstd.joinDirPathAndName(kstd.getScriptDir(),"_label1000.csv")
    
    dtoNL = kstd.DtoNpList()
    dtoNL.add(predict_y)

    kstd.writeNewCsvDataList(output_file_path,dtoNL)

    kstd.echoFinish(process)







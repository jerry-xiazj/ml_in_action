import adaboost
from numpy import *

datArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
classifierArray, aggClassEst = adaboost.adaBoostTrainDS(datArr, labelArr, 50)
# testArr, testLabelArr = adaboost.loadDataSet('horseColicTest2.txt')
# prediction10 = adaboost.adaClassify(testArr, classifierArray)
# errArr = mat(ones((67, 1)))
# errRate = errArr[prediction10 != mat(testLabelArr).T].sum()/67
# print(errRate)
adaboost.plotROC(aggClassEst.T, labelArr)

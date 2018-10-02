import random

import numpy as np
import math
import idx2numpy
from Q5.naive_bayes import NB
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


def create_test_train_set(train, test, ytrain, ytest, arr):
    ftrainx = []
    ftrainy = []
    ftestx = []
    ftesty = []
    print(len(train),len(test))
    for i in range(0, len(train)):
        if (ytrain[i] in arr):
            ftrainx.append(train[i])
            ftrainy.append(ytrain[i])

    for i in range(0, len(test)):
        if (ytest[i] in arr):
            ftestx.append(test[i])
            ftesty.append(ytest[i])

    ftrainx = np.array(ftrainx)
    ftestx = np.array(ftestx)

    return ftrainx, ftestx, ftrainy, ftesty


def createset(data,y,l,percent,label):
    f_x=[]
    # f_y=[]
    for i in range(0,len(data)):
        if(int(y[i])==l):
            f_x.append(data[i])
            # f_y.append(y[i])
    # print(len(f_x),"Length", l)
    # index = random.sample(range(0, len(f_x)), int(len(f_x) * percent))
    index=np.random.randint(len(f_x),size=int(len(f_x)*percent))
    # print(index)
    f_x=np.array(f_x)
    # final_x=[]
    # final_y=[]
    # for i in index:
    #     final_x.append(f_x[i])
    #     final_y.append(l)

    # # print("length",l ,len(final_x))
    return list(f_x[index,:]),[label]*len(index)







data_train = idx2numpy.convert_from_file('mnist/train/train-images.idx3-ubyte')
data_test = idx2numpy.convert_from_file('mnist/test/t10k-images.idx3-ubyte')
label_train = idx2numpy.convert_from_file('mnist/train/train-labels.idx1-ubyte')
label_test = idx2numpy.convert_from_file(('mnist/test/t10k-labels.idx1-ubyte'))
# print(label_train.shape)


data_train_flatten=[]
data_test_flatten=[]


for i in range(0,data_train.shape[0]):
    data_train_flatten.append(data_train[i].flatten())

for i in range(0,data_test.shape[0]):
    data_test_flatten.append(data_test[i].flatten())

data_test_flatten=np.array(data_test_flatten)
data_train_flatten=np.array(data_train_flatten)
print(data_train_flatten.shape)

#
train_3,ytrain_3 = createset(data_train_flatten,label_train,3,0.1,0)
# print(np.unique(np.array(ytrain_3),return_counts=True))
train_8,ytrain_8 = createset(data_train_flatten,label_train,8,0.9,1)
print(len(train_3),len(train_8))
trainx=train_3+train_8
ytrain=ytrain_3+ytrain_8
# # print(ytrain)
#
#
test_3,ytest_3=createset(data_test_flatten,label_test,3,0.1,0)
test_8,ytest_8 = createset(data_test_flatten,label_test,8,0.9,1)
testx=test_3+test_8
ytest=ytest_3+ytest_8
# # # print(len(ytest))

# trainx,testx,ytrain,ytest=create_test_train_set(data_train_flatten,data_test_flatten,label_train,label_test,[0,1])
# print(len(trainx),len(testx),len(ytrain),len(ytest))


model = NB()
model.fit(np.array(trainx),ytrain)
# model.fit(trainx,ytrain)
# Y=model.predict()

Y,score=model.predict(np.array(testx))
# Y,score=model.predict(testx)
score_list=[]

print(model.index)

for i in range(score.shape[1]):
    score_list.append(score[0][i]-score[1][i])
print(len(score_list))
# print(len(Y))
count=0
for i in range(0,len(Y)):
    if(Y[i]==ytest[i]):
        count=count+1
print(count/float(len(Y)))
#
print(np.unique(Y,return_counts=True))
#

# t = GaussianNB()
# t.fit(trainx,ytrain)
# print(t.score(testx,ytest))
fpr,tpr,_=roc_curve(ytest,score_list,pos_label=0)
plt.plot(fpr,tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()
# # print(Y)
# # for i in range(0,len())
#
#
#

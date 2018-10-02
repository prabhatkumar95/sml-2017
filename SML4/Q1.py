import numpy as np
import pandas as pd
from sklearn.externals import joblib

from sklearn.metrics import confusion_matrix,roc_curve
from sklearn.model_selection import train_test_split
from NN import NN
import matplotlib.pyplot as plt
import seaborn

def plot_conf(predicted,truth,filename,title):
    conf = confusion_matrix(truth, predicted, labels=[0,1])
    norm_conf = (conf.T / conf.sum(axis=1)).T
    df_cm = pd.DataFrame(norm_conf, index=[0,1],
                         columns=[0,1])
    plt.figure(figsize=(10, 7))
    ax = seaborn.heatmap(df_cm, annot=True)
    ax.set(xlabel='Ground Truth', ylabel='Predicted')
    plt.title(title)
    plt.savefig("plots/"+filename+".png")


# data = np.loadtxt("adult.data",delimiter=",",dtype=str)
# data = pd.read_csv("adult.data",header=None,delimiter=",")
# label = data[:,-1]
# data = data[:,:-1]
# print(np.unique(label))
# print(data.shape)
# print (data.columns.values.tolist())

# cols_to_transform = [ 1,3,5,6,7,8,9,13,14 ]
# data_temp = pd.get_dummies(data,columns=cols_to_transform)

# print(data/_temp.iloc[1])



data = joblib.load("x.data")
label = joblib.load("y.label")

trainx=[]
trainy=[]
testx=[]
testy=[]

for x in range(0,len(data)):
    if(x%2==0):
        trainx.append(data[x])
        trainy.append(label[x])
    else:
        testy.append(label[x])
        testx.append(data[x])


trainy=np.array(trainy)
trainx=np.array(trainx)
testx=np.array(testx)
testy=np.array(testy)

# print(data.shape)
# print(label.shape)

# trainx,testx,trainy,testy = train_test_split(data,label,test_size=0.5)
model=NN([3],0.005)
model.fit(trainx,trainy,10)
joblib.dump(model,"neural.sav")
acr,scr,pred=model.score(testx,testy)
print(acr)


plot_conf(np.array(pred),np.argmax(testy,axis=1),"conf_NN","NN Relu-Softmax")
plt.close()
fpr,tpr,thres=roc_curve(np.argmax(testy,axis=1),scr,pos_label=0)
plt.plot(fpr,tpr)
plt.savefig("plots/rocNN.png")
plt.close()



plt.plot(np.arange(0,len(fpr)),fpr,label="FPR")
plt.plot(np.arange(0,len(fpr)),np.subtract(1,np.array(tpr)),label="FNR")
plt.legend()
plt.savefig("plots/errNN.png")

